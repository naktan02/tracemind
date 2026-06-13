import { SegmentBuffer } from "./segmentBuffer";
import { readTextSurfaceSnapshot } from "./surfaceDetector";
import type {
  ChildSupportSuggestionPayload,
  TypingSegmentPayload,
} from "../contracts/generated";

declare const chrome: {
  runtime?: {
    sendMessage: (
      message: unknown,
      callback?: (response: unknown) => void,
    ) => void;
    onMessage?: {
      addListener: (callback: (message: unknown) => void) => void;
    };
  };
};

const DEFAULT_IDLE_MS = 5000;
const EDITOR_SETTLE_MS = 80;
const TYPING_SEGMENT_CAPTURED_MESSAGE = "tracemind.typingSegmentCaptured";
const COLLECTOR_CONTENT_STATUS_MESSAGE = "tracemind.collectorContentStatus";
const PROACTIVE_PROMPT_AVAILABLE_MESSAGE = "tracemind.proactivePromptAvailable";
const CHILD_SUPPORT_MESSAGE_REQUESTED_MESSAGE =
  "tracemind.childSupportMessageRequested";
const surfaceElementIds = new WeakMap<HTMLElement, string>();
let proactivePopupRoot: HTMLDivElement | null = null;
let proactiveConversationId: string | null = null;
let lastPromptText: string | null = null;

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
chrome.runtime?.onMessage?.addListener((message) => {
  if (!isProactivePromptAvailableMessage(message)) {
    return;
  }
  showProactiveCoachPopup({
    promptText: message.promptText,
    suggestedPrompts: message.suggestedPrompts,
  });
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

function showProactiveCoachPopup({
  promptText,
  suggestedPrompts,
}: {
  promptText: string;
  suggestedPrompts: ChildSupportSuggestionPayload[];
}): void {
  if (promptText.trim() === "" || promptText === lastPromptText) {
    return;
  }
  if (proactivePopupRoot !== null && document.contains(proactivePopupRoot)) {
    lastPromptText = promptText;
    return;
  }
  lastPromptText = promptText;
  proactiveConversationId = null;
  proactivePopupRoot?.remove();

  const root = document.createElement("div");
  root.id = "tracemind-proactive-coach";
  root.innerHTML = `
    <style>
      #tracemind-proactive-coach {
        position: fixed;
        inset: 0;
        z-index: 2147483647;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      #tracemind-proactive-coach .tm-card {
        width: min(440px, calc(100vw - 32px));
        max-height: min(640px, calc(100vh - 32px));
        display: flex;
        flex-direction: column;
        border: 1px solid #123d38;
        border-radius: 8px;
        background: #123d38;
        color: #f8faf7;
        box-shadow: 0 22px 70px rgba(17, 24, 39, 0.34);
        overflow: hidden;
        pointer-events: auto;
      }

      #tracemind-proactive-coach .tm-header {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 16px 16px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.14);
      }

      #tracemind-proactive-coach .tm-label {
        margin: 0 0 4px;
        color: #9ce4d9;
        font-size: 12px;
        font-weight: 850;
        letter-spacing: 0;
      }

      #tracemind-proactive-coach h2 {
        margin: 0;
        color: #f8faf7;
        font-size: 18px;
        line-height: 1.3;
      }

      #tracemind-proactive-coach .tm-close {
        width: 34px;
        height: 34px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.08);
        color: #f8faf7;
        font-size: 18px;
        font-weight: 900;
        cursor: pointer;
      }

      #tracemind-proactive-coach .tm-thread {
        display: grid;
        gap: 10px;
        max-height: 300px;
        overflow-y: auto;
        padding: 14px 16px 6px;
      }

      #tracemind-proactive-coach .tm-row {
        display: flex;
      }

      #tracemind-proactive-coach .tm-row.child {
        justify-content: flex-end;
      }

      #tracemind-proactive-coach .tm-bubble {
        max-width: 92%;
        border-left: 5px solid #9ce4d9;
        border-radius: 4px;
        background: #f7fbf8;
        color: #17202a;
        padding: 12px 13px;
        font-size: 14px;
        line-height: 1.55;
        white-space: pre-line;
      }

      #tracemind-proactive-coach .tm-row.child .tm-bubble {
        border-left-color: #e7b15a;
        background: #fff5df;
      }

      #tracemind-proactive-coach .tm-suggestions {
        display: flex;
        gap: 8px;
        overflow-x: auto;
        padding: 8px 16px 4px;
      }

      #tracemind-proactive-coach .tm-chip {
        flex-shrink: 0;
        min-height: 34px;
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 4px;
        background: transparent;
        color: #f8faf7;
        padding: 0 10px;
        font-size: 13px;
        font-weight: 800;
        cursor: pointer;
      }

      #tracemind-proactive-coach .tm-composer {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 8px;
        padding: 12px 16px 16px;
      }

      #tracemind-proactive-coach textarea {
        min-height: 48px;
        max-height: 130px;
        resize: vertical;
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 4px;
        background: #f8faf7;
        color: #17202a;
        padding: 12px 13px;
        font: inherit;
        font-size: 14px;
        outline: none;
      }

      #tracemind-proactive-coach .tm-send {
        min-width: 76px;
        border: 1px solid #9ce4d9;
        border-radius: 4px;
        background: #9ce4d9;
        color: #123d38;
        font: inherit;
        font-weight: 900;
        cursor: pointer;
      }

      #tracemind-proactive-coach .tm-error {
        margin: 8px 16px 0;
        border: 1px solid rgba(253, 186, 116, 0.42);
        border-radius: 4px;
        background: rgba(255, 237, 213, 0.14);
        color: #fed7aa;
        padding: 10px 12px;
        font-size: 13px;
      }
    </style>
    <section class="tm-card" role="dialog" aria-label="TraceMind AI 마음 도움">
      <div class="tm-header">
        <div>
          <p class="tm-label">TraceMind AI 마음 도움</p>
          <h2>잠깐 같이 확인해요</h2>
        </div>
        <button class="tm-close" type="button" aria-label="닫기">x</button>
      </div>
      <div class="tm-thread" aria-live="polite"></div>
      <div class="tm-suggestions" aria-label="추천 입력"></div>
      <div class="tm-error" hidden></div>
      <form class="tm-composer">
        <textarea rows="1" placeholder="지금 느끼는 걸 짧게 적어도 괜찮아요"></textarea>
        <button class="tm-send" type="submit">보내기</button>
      </form>
    </section>
  `;
  document.documentElement.appendChild(root);
  proactivePopupRoot = root;

  const thread = getPopupElement(root, ".tm-thread", HTMLDivElement);
  const suggestions = getPopupElement(root, ".tm-suggestions", HTMLDivElement);
  const form = getPopupElement(root, ".tm-composer", HTMLFormElement);
  const textarea = getPopupElement(root, "textarea", HTMLTextAreaElement);
  const error = getPopupElement(root, ".tm-error", HTMLDivElement);
  getPopupElement(root, ".tm-close", HTMLButtonElement).addEventListener(
    "click",
    () => {
      root.remove();
      if (proactivePopupRoot === root) {
        proactivePopupRoot = null;
      }
    },
  );

  appendCoachBubble(thread, "assistant", promptText);
  renderSuggestions(suggestions, suggestedPrompts, (prompt) => {
    textarea.value = prompt;
    void submitProactiveCoachMessage(root, thread, textarea, error);
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void submitProactiveCoachMessage(root, thread, textarea, error);
  });
  textarea.focus();
}

async function submitProactiveCoachMessage(
  root: HTMLElement,
  thread: HTMLElement,
  textarea: HTMLTextAreaElement,
  error: HTMLDivElement,
): Promise<void> {
  const message = textarea.value.trim();
  if (message === "") {
    return;
  }
  textarea.value = "";
  setPopupError(error, null);
  appendCoachBubble(thread, "child", message);
  const waiting = appendCoachBubble(thread, "assistant", "응답을 조심스럽게 고르는 중...");
  const response = await requestChildSupportMessage(message);
  waiting.remove();
  if (!response.ok) {
    setPopupError(error, response.errorMessage);
    return;
  }
  proactiveConversationId = response.response.conversation_id;
  appendCoachBubble(thread, "assistant", response.response.reply_text);
  const suggestions = root.querySelector(".tm-suggestions");
  if (suggestions instanceof HTMLDivElement) {
    renderSuggestions(suggestions, response.response.suggested_prompts, (prompt) => {
      textarea.value = prompt;
      void submitProactiveCoachMessage(root, thread, textarea, error);
    });
  }
}

function requestChildSupportMessage(
  message: string,
): Promise<ChildSupportMessageResponse> {
  return new Promise((resolve) => {
    chrome.runtime?.sendMessage(
      {
        type: CHILD_SUPPORT_MESSAGE_REQUESTED_MESSAGE,
        message,
        conversationId: proactiveConversationId,
      },
      (response) => {
        if (isChildSupportMessageResponse(response)) {
          resolve(response);
          return;
        }
        resolve({
          ok: false,
          errorMessage: "AI 마음 도움 응답을 아직 받지 못했습니다.",
        });
      },
    );
  });
}

function appendCoachBubble(
  thread: HTMLElement,
  role: "assistant" | "child",
  text: string,
): HTMLDivElement {
  const row = document.createElement("div");
  row.className = role === "child" ? "tm-row child" : "tm-row assistant";
  const bubble = document.createElement("div");
  bubble.className = "tm-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  thread.appendChild(row);
  thread.scrollTop = thread.scrollHeight;
  return row;
}

function renderSuggestions(
  container: HTMLElement,
  suggestions: readonly ChildSupportSuggestionPayload[],
  onSelect: (prompt: string) => void,
): void {
  container.textContent = "";
  for (const suggestion of suggestions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tm-chip";
    button.textContent = suggestion.label;
    button.addEventListener("click", () => onSelect(suggestion.prompt));
    container.appendChild(button);
  }
}

function setPopupError(error: HTMLDivElement, message: string | null): void {
  if (message === null) {
    error.hidden = true;
    error.textContent = "";
    return;
  }
  error.hidden = false;
  error.textContent = message;
}

function getPopupElement<T extends HTMLElement>(
  root: HTMLElement,
  selector: string,
  constructor: { new (...args: never[]): T },
): T {
  const element = root.querySelector(selector);
  if (!(element instanceof constructor)) {
    throw new Error(`TraceMind popup element not found: ${selector}`);
  }
  return element;
}

function isChildSupportMessageResponse(
  value: unknown,
): value is ChildSupportMessageResponse {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ChildSupportMessageResponse>;
  return typeof candidate.ok === "boolean";
}

type ProactivePromptAvailableMessage = {
  type: typeof PROACTIVE_PROMPT_AVAILABLE_MESSAGE;
  promptText: string;
  suggestedPrompts: ChildSupportSuggestionPayload[];
};

type ChildSupportMessageResponse =
  | {
      ok: true;
      response: {
        conversation_id: string;
        reply_text: string;
        suggested_prompts: ChildSupportSuggestionPayload[];
      };
    }
  | {
      ok: false;
      errorMessage: string;
    };

function isProactivePromptAvailableMessage(
  value: unknown,
): value is ProactivePromptAvailableMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ProactivePromptAvailableMessage>;
  return (
    candidate.type === PROACTIVE_PROMPT_AVAILABLE_MESSAGE &&
    typeof candidate.promptText === "string" &&
    Array.isArray(candidate.suggestedPrompts)
  );
}
