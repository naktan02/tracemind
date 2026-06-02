import { SegmentBuffer } from "./segmentBuffer";
import { readTextSurfaceSnapshot } from "./surfaceDetector";
import type { TypingSegmentPayload } from "../contracts/generated";

declare const chrome: {
  runtime?: {
    sendMessage: (message: unknown) => void;
  };
};

const ELEMENT_ID_ATTRIBUTE = "data-tracemind-surface-id";
const DEFAULT_IDLE_MS = 5000;
const TYPING_SEGMENT_CAPTURED_MESSAGE = "tracemind.typingSegmentCaptured";

const segmentBuffer = new SegmentBuffer(
  {
    idleMs: DEFAULT_IDLE_MS,
    sourceType: "browser_extension",
  },
  (segment) => sendSegment(segment),
);

document.addEventListener("beforeinput", handleInputLikeEvent, true);
document.addEventListener("input", handleInputLikeEvent, true);
document.addEventListener("compositionstart", handleInputLikeEvent, true);
document.addEventListener("compositionend", handleInputLikeEvent, true);
document.addEventListener("keydown", handleKeydownFlushEvent, true);
document.addEventListener("search", handleSurfaceFlushEvent, true);
document.addEventListener("submit", handleSubmitFlushEvent, true);
window.addEventListener("pagehide", () => segmentBuffer.flushAll());
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    segmentBuffer.flushAll();
  }
});

function handleInputLikeEvent(event: InputEvent | CompositionEvent | Event): void {
  const surface = readTextSurfaceSnapshot(event.target);
  if (surface === null) {
    return;
  }
  segmentBuffer.observe({
    elementId: getStableElementId(surface.element),
    snapshot: surface.snapshot,
    now: new Date(),
    inputType: readInputType(event),
    insertedText: readInsertedText(event),
    isCompositionUpdate: isCompositionUpdateEvent(event),
    locale: document.documentElement.lang || navigator.language || "ko",
  });
}

function handleKeydownFlushEvent(event: KeyboardEvent): void {
  if (event.key !== "Enter" || event.isComposing) {
    return;
  }
  flushEventSurface(event);
}

function handleSurfaceFlushEvent(event: Event): void {
  flushEventSurface(event);
}

function handleSubmitFlushEvent(_event: Event): void {
  segmentBuffer.flushAll();
}

function flushEventSurface(event: Event): void {
  const surface = readTextSurfaceSnapshot(event.target);
  if (surface === null) {
    return;
  }
  segmentBuffer.flushElement(getStableElementId(surface.element));
}

function getStableElementId(element: HTMLElement): string {
  const existing = element.getAttribute(ELEMENT_ID_ATTRIBUTE);
  if (existing !== null && existing !== "") {
    return existing;
  }
  const nextId = `surface_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  element.setAttribute(ELEMENT_ID_ATTRIBUTE, nextId);
  return nextId;
}

function readInputType(event: Event): string | null {
  return event instanceof InputEvent ? event.inputType : null;
}

function readInsertedText(event: Event): string | null {
  if (!(event instanceof InputEvent)) {
    return null;
  }
  return event.data;
}

function isCompositionUpdateEvent(event: Event): boolean {
  if (event instanceof CompositionEvent) {
    return true;
  }
  if (!(event instanceof InputEvent)) {
    return false;
  }
  return event.isComposing || (event.inputType ?? "").includes("Composition");
}

function sendSegment(segment: TypingSegmentPayload): void {
  chrome.runtime?.sendMessage({
    type: TYPING_SEGMENT_CAPTURED_MESSAGE,
    segment,
  });
}
