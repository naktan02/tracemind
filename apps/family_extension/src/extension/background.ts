import { buildAgentApiUrl } from "../common/agentClient";
import type {
  CapturedTextBatchIngestRequestPayload,
  CapturedTextEventPayload,
  CapturedTextSurfaceType,
  TypingSegmentPayload,
} from "../contracts/generated";
import {
  isCollectorContentStatusMessage,
  isTypingSegmentCapturedMessage,
} from "./messages";
import {
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
) => void;

type StorageGetCallback = (items: Record<string, unknown>) => void;

declare const chrome: {
  runtime: {
    onMessage: {
      addListener: (callback: RuntimeMessageCallback) => void;
    };
  };
  storage: {
    local: {
      get: (keys: string[], callback: StorageGetCallback) => void;
      set: (items: Record<string, unknown>, callback?: () => void) => void;
    };
  };
};

let isFlushing = false;
const MAX_CAPTURED_TEXT_BATCH_SIZE = 100;

chrome.runtime.onMessage.addListener((message) => {
  if (isCollectorContentStatusMessage(message)) {
    void saveStatusPatch(message.status);
    return;
  }
  if (!isTypingSegmentCapturedMessage(message)) {
    return;
  }
  void enqueueSegment(message.segment);
});

void flushQueue();

async function enqueueSegment(segment: TypingSegmentPayload): Promise<void> {
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
  void flushQueue();
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
    throw new Error(`Agent captured text ingest failed: ${response.status}`);
  }
}

async function loadQueue(): Promise<CapturedTextEventPayload[]> {
  const items = await storageGet([PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY]);
  const rawQueue = items[PENDING_CAPTURED_TEXT_EVENTS_STORAGE_KEY];
  return Array.isArray(rawQueue) ? (rawQueue as CapturedTextEventPayload[]) : [];
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
  if (items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] !== true) {
    return;
  }
  const rawHistory = items[TYPING_SEGMENT_HISTORY_STORAGE_KEY];
  const history = Array.isArray(rawHistory)
    ? (rawHistory as TypingSegmentPayload[])
    : [];
  await storageSet({
    [LAST_TYPING_SEGMENT_STORAGE_KEY]: segment,
    [TYPING_SEGMENT_HISTORY_STORAGE_KEY]: [segment, ...history].slice(0, 20),
  });
}

function segmentToCapturedTextEvent(
  segment: TypingSegmentPayload,
): CapturedTextEventPayload {
  return {
    schema_version: "captured_text_event.v1",
    event_id: segment.segment_id,
    occurred_at: segment.ended_at,
    text: readSegmentAnalysisText(segment),
    locale: segment.locale,
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
