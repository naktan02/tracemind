import { SegmentBuffer } from "./segmentBuffer";
import { readTextSurfaceSnapshot } from "./surfaceDetector";
import type { TypingSegmentPayload } from "../contracts/generated";

declare const chrome: {
  runtime?: {
    sendMessage: (message: unknown) => void;
  };
};

const DEFAULT_IDLE_MS = 5000;
const EDITOR_SETTLE_MS = 80;
const TYPING_SEGMENT_CAPTURED_MESSAGE = "tracemind.typingSegmentCaptured";
const COLLECTOR_CONTENT_STATUS_MESSAGE = "tracemind.collectorContentStatus";
const surfaceElementIds = new WeakMap<HTMLElement, string>();

const segmentBuffer = new SegmentBuffer(
  {
    idleMs: DEFAULT_IDLE_MS,
    sourceType: "browser_extension",
  },
  (segment) => sendSegment(segment),
);

type DeferredInputObservation = {
  eventType: string;
  inputType: string | null;
  insertedText: string | null;
  isCompositionUpdate: boolean;
  isLineBreakCommit: boolean;
  locale: string;
  observedAt: Date;
  target: EventTarget | null;
  path: EventTarget[];
  targetDescription: string;
};

document.addEventListener("input", handleInputLikeEvent, false);
document.addEventListener("compositionend", handleInputLikeEvent, false);
document.addEventListener("search", handleSurfaceFlushEvent, false);
document.addEventListener("submit", handleSubmitFlushEvent, false);
window.addEventListener("pagehide", () => segmentBuffer.flushAll());
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    segmentBuffer.flushAll();
  }
});

sendCollectorStatus({
  last_content_script_at: new Date().toISOString(),
  page_origin: window.location.origin,
  page_url: window.location.href,
});

function handleInputLikeEvent(event: InputEvent | CompositionEvent | Event): void {
  const observation: DeferredInputObservation = {
    eventType: event.type,
    inputType: readInputType(event),
    insertedText: readInsertedText(event),
    isCompositionUpdate: isCompositionUpdateEvent(event),
    isLineBreakCommit: isLineBreakCommitEvent(event),
    locale: document.documentElement.lang || navigator.language || "ko",
    observedAt: new Date(),
    target: event.target,
    path: event.composedPath(),
    targetDescription: event.type,
  };
  window.setTimeout(
    () => observeDeferredInput(observation),
    readDeferredObservationDelayMs(event),
  );
}

function observeDeferredInput(observation: DeferredInputObservation): void {
  const surface = readDeferredTextSurfaceSnapshot(observation);
  if (surface === null) {
    sendCollectorStatus({
      last_unmatched_input_at: new Date().toISOString(),
      last_unmatched_target: observation.targetDescription,
      page_origin: window.location.origin,
      page_url: window.location.href,
    });
    return;
  }
  const elementId = getStableElementId(surface.element);
  segmentBuffer.observe({
    elementId,
    snapshot: surface.snapshot,
    now: observation.observedAt,
    eventType: observation.eventType,
    inputType: observation.inputType,
    insertedText: observation.insertedText,
    isCompositionUpdate: observation.isCompositionUpdate,
    locale: observation.locale,
  });
  sendCollectorStatus({
    last_surface_observed_at: new Date().toISOString(),
    last_surface_type: surface.snapshot.surfaceType,
    last_field_hint: surface.snapshot.fieldHint,
    page_origin: window.location.origin,
    page_url: window.location.href,
  });
  if (observation.isLineBreakCommit) {
    segmentBuffer.flushElement(elementId);
  }
}

function handleSurfaceFlushEvent(event: Event): void {
  flushEventSurface(event);
}

function handleSubmitFlushEvent(_event: Event): void {
  segmentBuffer.flushAll();
}

function flushEventSurface(event: Event): void {
  const surface = readEventTextSurfaceSnapshot(event);
  if (surface === null) {
    sendCollectorStatus({
      last_unmatched_flush_at: new Date().toISOString(),
      last_unmatched_target: describeEventTarget(event),
      page_origin: window.location.origin,
      page_url: window.location.href,
    });
    return;
  }
  segmentBuffer.flushElement(getStableElementId(surface.element));
}

function readEventTextSurfaceSnapshot(event: Event): ReturnType<
  typeof readTextSurfaceSnapshot
> {
  const directSurface = readTextSurfaceSnapshot(event.target);
  if (directSurface !== null) {
    return directSurface;
  }
  for (const candidate of event.composedPath()) {
    const surface = readTextSurfaceSnapshot(candidate);
    if (surface !== null) {
      return surface;
    }
  }
  return null;
}

function readDeferredTextSurfaceSnapshot(
  observation: DeferredInputObservation,
): ReturnType<typeof readTextSurfaceSnapshot> {
  const directSurface = readTextSurfaceSnapshot(observation.target);
  if (directSurface !== null) {
    return directSurface;
  }
  for (const candidate of observation.path) {
    const surface = readTextSurfaceSnapshot(candidate);
    if (surface !== null) {
      return surface;
    }
  }
  return null;
}

function describeEventTarget(event: Event): string {
  const parts = event
    .composedPath()
    .slice(0, 6)
    .map((target) => {
      if (!(target instanceof Element)) {
        return target.constructor.name;
      }
      const tag = target.tagName.toLowerCase();
      const id = target.id ? `#${target.id}` : "";
      const className =
        typeof target.className === "string" && target.className.trim() !== ""
          ? `.${target.className.trim().split(/\s+/).slice(0, 3).join(".")}`
          : "";
      const role = target.getAttribute("role");
      return role === null ? `${tag}${id}${className}` : `${tag}${id}[${role}]`;
    });
  return `${event.type}: ${parts.join(" > ")}`.slice(0, 512);
}

function getStableElementId(element: HTMLElement): string {
  const existing = surfaceElementIds.get(element);
  if (existing !== undefined) {
    return existing;
  }
  const nextId = `surface_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  surfaceElementIds.set(element, nextId);
  return nextId;
}

function readInputType(event: Event): string | null {
  return event instanceof InputEvent ? event.inputType : null;
}

function readInsertedText(event: Event): string | null {
  if (event instanceof InputEvent) {
    return event.data;
  }
  if (event instanceof CompositionEvent) {
    return event.data || null;
  }
  return null;
}

function isCompositionUpdateEvent(event: Event): boolean {
  if (!(event instanceof InputEvent)) {
    return false;
  }
  return event.isComposing;
}

function isLineBreakCommitEvent(event: Event): boolean {
  if (!(event instanceof InputEvent) || event.isComposing) {
    return false;
  }
  return (
    event.inputType === "insertParagraph" ||
    event.inputType === "insertLineBreak"
  );
}

function readDeferredObservationDelayMs(event: Event): number {
  if (event instanceof CompositionEvent && event.type === "compositionend") {
    return EDITOR_SETTLE_MS;
  }
  if (
    event instanceof InputEvent &&
    !event.isComposing &&
    event.inputType.includes("Composition")
  ) {
    return EDITOR_SETTLE_MS;
  }
  return 0;
}

function sendSegment(segment: TypingSegmentPayload): void {
  sendCollectorStatus({
    last_flush_attempt_at: new Date().toISOString(),
    last_flush_surface_type: segment.surface_type,
    last_flush_text_length: segment.final_text?.length ?? 0,
    page_origin: window.location.origin,
    page_url: window.location.href,
  });
  chrome.runtime?.sendMessage({
    type: TYPING_SEGMENT_CAPTURED_MESSAGE,
    segment,
  });
}

function sendCollectorStatus(status: Record<string, unknown>): void {
  chrome.runtime?.sendMessage({
    type: COLLECTOR_CONTENT_STATUS_MESSAGE,
    status,
  });
}
