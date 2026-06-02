import { buildAgentApiUrl } from "../common/agentClient";
import type { TypingSegmentPayload } from "../contracts/generated";
import { isTypingSegmentCapturedMessage } from "./messages";
import {
  COLLECTOR_DEBUG_ENABLED_STORAGE_KEY,
  COLLECTOR_STATUS_STORAGE_KEY,
  LAST_TYPING_SEGMENT_STORAGE_KEY,
  PENDING_TYPING_SEGMENTS_STORAGE_KEY,
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

chrome.runtime.onMessage.addListener((message) => {
  if (!isTypingSegmentCapturedMessage(message)) {
    return;
  }
  void enqueueSegment(message.segment);
});

void flushQueue();

async function enqueueSegment(segment: TypingSegmentPayload): Promise<void> {
  const queue = await loadQueue();
  queue.push(segment);
  await saveQueue(queue);
  await saveLastSegmentForDebug(segment);
  await saveStatus({
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
      const [nextSegment, ...remaining] = queue;
      await postSegment(nextSegment);
      queue = remaining;
      await saveQueue(queue);
      await saveStatus({
        pending_count: queue.length,
        last_sent_at: nextSegment.ended_at,
        last_error: null,
      });
    }
  } catch (error) {
    await saveStatus({
      pending_count: (await loadQueue()).length,
      last_error: error instanceof Error ? error.message : "segment 전송 실패",
    });
  } finally {
    isFlushing = false;
  }
}

async function postSegment(segment: TypingSegmentPayload): Promise<void> {
  const response = await fetch(buildAgentApiUrl("/api/v1/typing-segments"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(segment),
  });
  if (!response.ok) {
    throw new Error(`Agent typing segment ingest failed: ${response.status}`);
  }
}

async function loadQueue(): Promise<TypingSegmentPayload[]> {
  const items = await storageGet([PENDING_TYPING_SEGMENTS_STORAGE_KEY]);
  const rawQueue = items[PENDING_TYPING_SEGMENTS_STORAGE_KEY];
  return Array.isArray(rawQueue) ? (rawQueue as TypingSegmentPayload[]) : [];
}

function saveQueue(queue: TypingSegmentPayload[]): Promise<void> {
  return storageSet({ [PENDING_TYPING_SEGMENTS_STORAGE_KEY]: queue });
}

function saveStatus(status: Record<string, unknown>): Promise<void> {
  return storageSet({ [COLLECTOR_STATUS_STORAGE_KEY]: status });
}

async function saveLastSegmentForDebug(
  segment: TypingSegmentPayload,
): Promise<void> {
  const items = await storageGet([COLLECTOR_DEBUG_ENABLED_STORAGE_KEY]);
  if (items[COLLECTOR_DEBUG_ENABLED_STORAGE_KEY] !== true) {
    return;
  }
  await storageSet({ [LAST_TYPING_SEGMENT_STORAGE_KEY]: segment });
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
