import { buildAgentApiUrl } from "../common/agentClient";
import type {
  CapturedTextBatchIngestRequestPayload,
  CapturedTextDebugJobRunRequestPayload,
  CapturedTextEventPayload,
  CapturedTextSurfaceType,
  ChildSupportConversationRequestPayload,
  ChildSupportConversationResponsePayload,
  ChildSupportProactivePromptPayload,
  TypingSegmentPayload,
} from "../contracts/generated";
import {
  PROACTIVE_PROMPT_AVAILABLE_MESSAGE,
  type ChildSupportMessageResponse,
  isChildSupportMessageRequestedMessage,
  isCollectorContentStatusMessage,
  isProactivePromptDismissedMessage,
  isTypingSegmentCapturedMessage,
} from "./messages";
import {
  CHILD_SUPPORT_PROACTIVE_DISMISSED_UNTIL_STORAGE_KEY,
  CHILD_SUPPORT_PROACTIVE_TAB_IDS_STORAGE_KEY,
  COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY,
  COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
  COLLECTOR_STATUS_STORAGE_KEY,
  LAST_TYPING_SEGMENT_STORAGE_KEY,
  PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY,
  TYPING_SEGMENT_HISTORY_STORAGE_KEY,
} from "./storageKeys";

type RuntimeMessageSender = {
  tab?: {
    id?: number;
  };
};

type RuntimeMessageCallback = (
  message: unknown,
  sender: RuntimeMessageSender,
  sendResponse: (response: unknown) => void,
) => boolean | void;

type StorageGetCallback = (items: Record<string, unknown>) => void;

declare const chrome: {
  runtime: {
    lastError?: {
      message?: string;
    };
    onMessage: {
      addListener: (callback: RuntimeMessageCallback) => void;
    };
    onInstalled?: {
      addListener: (callback: () => void) => void;
    };
    onStartup?: {
      addListener: (callback: () => void) => void;
    };
  };
  storage: {
    local: {
      get: (keys: string[], callback: StorageGetCallback) => void;
      set: (items: Record<string, unknown>, callback?: () => void) => void;
    };
  };
  tabs?: {
    sendMessage: (
      tabId: number,
      message: unknown,
      callback?: () => void,
    ) => void;
  };
  alarms?: {
    create: (
      name: string,
      alarmInfo: { delayInMinutes?: number; periodInMinutes: number },
    ) => void;
    onAlarm: {
      addListener: (callback: (alarm: { name: string }) => void) => void;
    };
  };
};

let isFlushing = false;
let isDebugPipelineRunning = false;
const MAX_CAPTURED_TEXT_BATCH_SIZE = 100;
const DEBUG_PIPELINE_BATCH_LIMIT = 20;
const MAX_PROACTIVE_CONTENT_TABS = 12;
const PROACTIVE_PROMPT_ALARM_NAME = "tracemind.childSupportPromptPoll";
const PROACTIVE_PROMPT_POLL_MINUTES = 0.5;
const PROACTIVE_PROMPT_DISMISS_COOLDOWN_MS = 30 * 60 * 1000;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (isCollectorContentStatusMessage(message)) {
    void rememberContentTab(sender.tab?.id);
    void saveStatusPatch(message.status);
    return;
  }
  if (isChildSupportMessageRequestedMessage(message)) {
    void rememberContentTab(sender.tab?.id);
    void postChildSupportMessage(message)
      .then((response) => sendResponse(response))
      .catch((error) =>
        sendResponse({
          ok: false,
          errorMessage:
            error instanceof Error ? error.message : "AI 마음 도움 응답 실패",
        } satisfies ChildSupportMessageResponse),
    );
    return true;
  }
  if (isProactivePromptDismissedMessage(message)) {
    void saveProactivePromptDismissal();
    return;
  }
  if (!isTypingSegmentCapturedMessage(message)) {
    return;
  }
  void rememberContentTab(sender.tab?.id);
  void enqueueSegment(message.segment, sender.tab?.id);
});

void flushQueue();
startProactivePromptMonitor();

async function enqueueSegment(
  segment: TypingSegmentPayload,
  sourceTabId?: number,
): Promise<void> {
  const event = segmentToCapturedTextEvent(segment);
  const queue = await loadQueue();
  queue.push(event);
  await saveQueue(queue);
  await saveLastSegmentForDebug(segment);
  await saveStatusPatch({
    pending_count: queue.length,
    last_segment_at: segment.ended_at,
    last_error: null,
  });
  await flushQueue();
  await runDebugPipelineForCapturedInput(sourceTabId);
  void pollProactivePrompt({ preferredTabId: sourceTabId });
}

async function flushQueue(): Promise<void> {
  if (isFlushing) {
    return;
  }
  isFlushing = true;
  try {
    let queue = await loadQueue();
    while (queue.length > 0) {
      const batch = queue.slice(0, MAX_CAPTURED_TEXT_BATCH_SIZE);
      await postCapturedTextBatch(batch);
      queue = queue.slice(batch.length);
      await saveQueue(queue);
      await saveStatusPatch({
        pending_count: queue.length,
        last_sent_at: batch[batch.length - 1]?.occurred_at ?? null,
        last_error: null,
      });
    }
  } catch (error) {
    await saveStatusPatch({
      pending_count: (await loadQueue()).length,
      last_error: error instanceof Error ? error.message : "segment 전송 실패",
    });
  } finally {
    isFlushing = false;
  }
}

async function runDebugPipelineForCapturedInput(
  sourceTabId: number | undefined,
): Promise<void> {
  if (!(await loadDebugPipelineEnabled()) || isDebugPipelineRunning) {
    return;
  }
  isDebugPipelineRunning = true;
  try {
    await runCapturedTextDebugStep("/debug-job/run-view-generation");
    await runCapturedTextDebugStep("/debug-job/run-analysis");
    await saveStatusPatch({
      last_debug_pipeline_at: new Date().toISOString(),
      last_debug_pipeline_error: null,
    });
    await pollProactivePrompt({ preferredTabId: sourceTabId });
  } catch (error) {
    await saveStatusPatch({
      last_debug_pipeline_error:
        error instanceof Error ? error.message : "debug pipeline 실행 실패",
    });
  } finally {
    isDebugPipelineRunning = false;
  }
}

async function runCapturedTextDebugStep(path: string): Promise<void> {
  const payload: CapturedTextDebugJobRunRequestPayload = {
    limit: DEBUG_PIPELINE_BATCH_LIMIT,
  };
  const response = await fetch(buildAgentApiUrl(`/api/v1/captured-text${path}`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await readAgentErrorDetail(response);
    throw new Error(
      detail === null
        ? `Agent debug pipeline failed: ${response.status}`
        : `Agent debug pipeline failed: ${response.status}: ${detail}`,
    );
  }
}

function startProactivePromptMonitor(): void {
  chrome.alarms?.onAlarm.addListener((alarm) => {
    if (alarm.name !== PROACTIVE_PROMPT_ALARM_NAME) {
      return;
    }
    void pollProactivePrompt();
  });
  chrome.alarms?.create(PROACTIVE_PROMPT_ALARM_NAME, {
    delayInMinutes: PROACTIVE_PROMPT_POLL_MINUTES,
    periodInMinutes: PROACTIVE_PROMPT_POLL_MINUTES,
  });
  chrome.runtime.onInstalled?.addListener(() => {
    chrome.alarms?.create(PROACTIVE_PROMPT_ALARM_NAME, {
      delayInMinutes: PROACTIVE_PROMPT_POLL_MINUTES,
      periodInMinutes: PROACTIVE_PROMPT_POLL_MINUTES,
    });
  });
  chrome.runtime.onStartup?.addListener(() => {
    chrome.alarms?.create(PROACTIVE_PROMPT_ALARM_NAME, {
      delayInMinutes: PROACTIVE_PROMPT_POLL_MINUTES,
      periodInMinutes: PROACTIVE_PROMPT_POLL_MINUTES,
    });
  });
  void pollProactivePrompt();
}

async function pollProactivePrompt({
  preferredTabId,
}: {
  preferredTabId?: number;
} = {}): Promise<void> {
  const checkedAt = new Date().toISOString();
  try {
    if (chrome.tabs === undefined) {
      await saveStatusPatch({
        last_proactive_prompt_checked_at: checkedAt,
        last_proactive_prompt_error: "chrome.tabs API를 사용할 수 없습니다.",
      });
      return;
    }
    const response = await fetch(
      buildAgentApiUrl("/api/v1/child-support/proactive-prompt"),
    );
    if (!response.ok) {
      await saveStatusPatch({
        last_proactive_prompt_checked_at: checkedAt,
        last_proactive_prompt_error: `proactive prompt failed: ${response.status}`,
      });
      return;
    }
    const prompt = (await response.json()) as ChildSupportProactivePromptPayload;
    if (!prompt.should_prompt || prompt.prompt_text === null) {
      await saveStatusPatch({
        last_proactive_prompt_checked_at: checkedAt,
        last_proactive_prompt_should_prompt: false,
        last_proactive_prompt_error: null,
        last_proactive_prompt_suppressed_by_dismissal: false,
      });
      return;
    }
    if (await isProactivePromptDismissed()) {
      await saveStatusPatch({
        last_proactive_prompt_checked_at: checkedAt,
        last_proactive_prompt_should_prompt: true,
        last_proactive_prompt_delivered_count: 0,
        last_proactive_prompt_error: null,
        last_proactive_prompt_suppressed_by_dismissal: true,
      });
      return;
    }
    const tabIds = await loadContentTabIds(preferredTabId);
    if (tabIds.length === 0) {
      await saveStatusPatch({
        last_proactive_prompt_checked_at: checkedAt,
        last_proactive_prompt_should_prompt: true,
        last_proactive_prompt_delivered_count: 0,
        last_proactive_prompt_error: "팝업을 보낼 content tab이 없습니다.",
      });
      return;
    }
    const message = {
      type: PROACTIVE_PROMPT_AVAILABLE_MESSAGE,
      conversationId: prompt.conversation_id,
      promptText: prompt.prompt_text,
      suggestedPrompts: prompt.suggested_prompts,
    };
    let deliveredCount = 0;
    const failedTabIds: number[] = [];
    for (const tabId of tabIds) {
      if (await sendTabMessage(tabId, message)) {
        deliveredCount += 1;
      } else {
        failedTabIds.push(tabId);
      }
    }
    if (failedTabIds.length > 0) {
      await forgetContentTabs(failedTabIds);
    }
    await saveStatusPatch({
      last_proactive_prompt_checked_at: checkedAt,
      last_proactive_prompt_should_prompt: true,
      last_proactive_prompt_delivered_count: deliveredCount,
      last_proactive_prompt_error:
        deliveredCount > 0 ? null : "content script가 응답한 탭이 없습니다.",
      last_proactive_prompt_suppressed_by_dismissal: false,
    });
  } catch (error) {
    await saveStatusPatch({
      last_proactive_prompt_checked_at: checkedAt,
      last_proactive_prompt_error:
        error instanceof Error ? error.message : "proactive prompt 확인 실패",
    });
  }
}

async function postChildSupportMessage(
  message: {
    message: string;
    conversationId: string | null;
  },
): Promise<ChildSupportMessageResponse> {
  const payload: ChildSupportConversationRequestPayload = {
    message: message.message,
    conversation_id: message.conversationId,
  };
  const response = await fetch(buildAgentApiUrl("/api/v1/child-support/messages"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await readAgentErrorDetail(response);
    return {
      ok: false,
      errorMessage:
        detail === null
          ? `AI 마음 도움 응답 실패: ${response.status}`
          : `AI 마음 도움 응답 실패: ${response.status}: ${detail}`,
    };
  }
  return {
    ok: true,
    response: (await response.json()) as ChildSupportConversationResponsePayload,
  };
}

async function postCapturedTextBatch(
  events: CapturedTextEventPayload[],
): Promise<void> {
  const payload: CapturedTextBatchIngestRequestPayload = { events };
  const response = await fetch(buildAgentApiUrl("/api/v1/captured-text/batch"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await readAgentErrorDetail(response);
    throw new Error(
      detail === null
        ? `Agent captured text ingest failed: ${response.status}`
        : `Agent captured text ingest failed: ${response.status}: ${detail}`,
    );
  }
}

async function readAgentErrorDetail(response: Response): Promise<string | null> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" && payload.detail.trim() !== ""
      ? payload.detail
      : null;
  } catch {
    return null;
  }
}

async function loadQueue(): Promise<CapturedTextEventPayload[]> {
  const items = await storageGet([PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY]);
  const rawQueue = items[PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY];
  return Array.isArray(rawQueue) ? (rawQueue as CapturedTextEventPayload[]) : [];
}

async function rememberContentTab(tabId: number | undefined): Promise<void> {
  if (tabId === undefined) {
    return;
  }
  const existing = await loadContentTabIds();
  await saveContentTabIds([tabId, ...existing.filter((id) => id !== tabId)]);
}

async function loadContentTabIds(preferredTabId?: number): Promise<number[]> {
  const items = await storageGet([CHILD_SUPPORT_PROACTIVE_TAB_IDS_STORAGE_KEY]);
  const rawIds = items[CHILD_SUPPORT_PROACTIVE_TAB_IDS_STORAGE_KEY];
  const storedIds = Array.isArray(rawIds)
    ? rawIds.filter((id): id is number => Number.isInteger(id))
    : [];
  if (preferredTabId === undefined) {
    return storedIds;
  }
  return [preferredTabId, ...storedIds.filter((id) => id !== preferredTabId)];
}

async function forgetContentTabs(tabIds: number[]): Promise<void> {
  if (tabIds.length === 0) {
    return;
  }
  const failed = new Set(tabIds);
  const existing = await loadContentTabIds();
  await saveContentTabIds(existing.filter((id) => !failed.has(id)));
}

function saveContentTabIds(tabIds: number[]): Promise<void> {
  return storageSet({
    [CHILD_SUPPORT_PROACTIVE_TAB_IDS_STORAGE_KEY]: tabIds.slice(
      0,
      MAX_PROACTIVE_CONTENT_TABS,
    ),
  });
}

async function saveProactivePromptDismissal(): Promise<void> {
  const dismissedUntil = new Date(
    Date.now() + PROACTIVE_PROMPT_DISMISS_COOLDOWN_MS,
  ).toISOString();
  await storageSet({
    [CHILD_SUPPORT_PROACTIVE_DISMISSED_UNTIL_STORAGE_KEY]: dismissedUntil,
  });
  await saveStatusPatch({
    last_proactive_prompt_dismissed_at: new Date().toISOString(),
    last_proactive_prompt_dismissed_until: dismissedUntil,
  });
}

async function isProactivePromptDismissed(): Promise<boolean> {
  const items = await storageGet([
    CHILD_SUPPORT_PROACTIVE_DISMISSED_UNTIL_STORAGE_KEY,
  ]);
  const dismissedUntil =
    items[CHILD_SUPPORT_PROACTIVE_DISMISSED_UNTIL_STORAGE_KEY];
  if (typeof dismissedUntil !== "string") {
    return false;
  }
  const dismissedUntilTime = Date.parse(dismissedUntil);
  return Number.isFinite(dismissedUntilTime) && Date.now() < dismissedUntilTime;
}

function sendTabMessage(tabId: number, message: unknown): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      chrome.tabs?.sendMessage(tabId, message, () => {
        resolve(chrome.runtime.lastError === undefined);
      });
    } catch {
      resolve(false);
    }
  });
}

async function loadDebugPipelineEnabled(): Promise<boolean> {
  const items = await storageGet([COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY]);
  return items[COLLECTOR_DEBUG_PIPELINE_ENABLED_STORAGE_KEY] === true;
}

function saveQueue(queue: CapturedTextEventPayload[]): Promise<void> {
  return storageSet({ [PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY]: queue });
}

async function saveStatusPatch(status: Record<string, unknown>): Promise<void> {
  const items = await storageGet([COLLECTOR_STATUS_STORAGE_KEY]);
  const previous = items[COLLECTOR_STATUS_STORAGE_KEY];
  const mergedStatus =
    typeof previous === "object" && previous !== null
      ? { ...(previous as Record<string, unknown>), ...status }
      : status;
  return storageSet({ [COLLECTOR_STATUS_STORAGE_KEY]: mergedStatus });
}

async function saveLastSegmentForDebug(
  segment: TypingSegmentPayload,
): Promise<void> {
  const items = await storageGet([
    COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
    TYPING_SEGMENT_HISTORY_STORAGE_KEY,
  ]);
  const rawHistory = items[TYPING_SEGMENT_HISTORY_STORAGE_KEY];
  const history = Array.isArray(rawHistory)
    ? (rawHistory as TypingSegmentPayload[])
    : [];
  if (items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] === true) {
    await storageSet({
      [LAST_TYPING_SEGMENT_STORAGE_KEY]: segment,
      [TYPING_SEGMENT_HISTORY_STORAGE_KEY]: [segment, ...history].slice(0, 20),
    });
    return;
  }
  await storageSet({ [LAST_TYPING_SEGMENT_STORAGE_KEY]: segment });
}

function segmentToCapturedTextEvent(
  segment: TypingSegmentPayload,
): CapturedTextEventPayload {
  const text = readSegmentAnalysisText(segment);
  return {
    schema_version: "captured_text_event.v1",
    event_id: segment.segment_id,
    occurred_at: segment.ended_at,
    text,
    locale: inferCapturedTextLocale(text, segment.locale),
    source_type: "typing",
    surface_type: "typing_segment",
    page_url: segment.page_url,
    page_title: null,
    collector_version: null,
    metadata: {
      producer_schema_version: segment.schema_version,
      producer_source_type: segment.source_type,
      producer_surface_type: segment.surface_type,
      producer_capture_confidence: segment.capture_confidence,
      captured_text_surface_type: mapTypingSurfaceToCapturedSurface(
        segment.surface_type,
      ),
      page_origin: segment.page_origin,
      field_hint: segment.field_hint,
      started_at: segment.started_at,
      idle_ms: segment.idle_ms,
      stats: segment.stats,
      used_text_field:
        (segment.final_text ?? "").trim() !== "" ? "final_text" : "deleted_text",
    },
  };
}

function inferCapturedTextLocale(text: string, fallbackLocale: string): string {
  if (/[\uac00-\ud7af\u3130-\u318f\u1100-\u11ff]/.test(text)) {
    return "ko";
  }
  return fallbackLocale;
}

function readSegmentAnalysisText(segment: TypingSegmentPayload): string {
  const finalText = (segment.final_text ?? "").trim();
  if (finalText !== "") {
    return finalText;
  }
  return (segment.deleted_text ?? "").trim();
}

function mapTypingSurfaceToCapturedSurface(
  surfaceType: TypingSegmentPayload["surface_type"],
): CapturedTextSurfaceType {
  if (surfaceType === "input") {
    return "search_box";
  }
  return "typing_segment";
}

function storageGet(keys: string[]): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    chrome.storage.local.get(keys, (items) => resolve(items));
  });
}

function storageSet(items: Record<string, unknown>): Promise<void> {
  return new Promise((resolve) => {
    chrome.storage.local.set(items, () => resolve());
  });
}
