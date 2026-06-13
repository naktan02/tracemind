import type {
  ChildSupportConversationResponsePayload,
  ChildSupportSuggestionPayload,
  TypingSegmentPayload,
} from "../contracts/generated";

export const TYPING_SEGMENT_CAPTURED_MESSAGE = "tracemind.typingSegmentCaptured";
export const COLLECTOR_CONTENT_STATUS_MESSAGE =
  "tracemind.collectorContentStatus";
export const PROACTIVE_PROMPT_AVAILABLE_MESSAGE =
  "tracemind.proactivePromptAvailable";
export const PROACTIVE_PROMPT_DISMISSED_MESSAGE =
  "tracemind.proactivePromptDismissed";
export const CHILD_SUPPORT_MESSAGE_REQUESTED_MESSAGE =
  "tracemind.childSupportMessageRequested";

export type TypingSegmentCapturedMessage = {
  type: typeof TYPING_SEGMENT_CAPTURED_MESSAGE;
  segment: TypingSegmentPayload;
};

export type CollectorContentStatusMessage = {
  type: typeof COLLECTOR_CONTENT_STATUS_MESSAGE;
  status: Record<string, unknown>;
};

export type ProactivePromptAvailableMessage = {
  type: typeof PROACTIVE_PROMPT_AVAILABLE_MESSAGE;
  conversationId: string | null;
  promptText: string;
  suggestedPrompts: ChildSupportSuggestionPayload[];
};

export type ProactivePromptDismissedMessage = {
  type: typeof PROACTIVE_PROMPT_DISMISSED_MESSAGE;
};

export type ChildSupportMessageRequestedMessage = {
  type: typeof CHILD_SUPPORT_MESSAGE_REQUESTED_MESSAGE;
  message: string;
  conversationId: string | null;
};

export type ChildSupportMessageResponse = {
  ok: true;
  response: ChildSupportConversationResponsePayload;
} | {
  ok: false;
  errorMessage: string;
};

export type ExtensionMessage =
  | TypingSegmentCapturedMessage
  | CollectorContentStatusMessage
  | ProactivePromptAvailableMessage
  | ProactivePromptDismissedMessage
  | ChildSupportMessageRequestedMessage;

export function isTypingSegmentCapturedMessage(
  value: unknown,
): value is TypingSegmentCapturedMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<TypingSegmentCapturedMessage>;
  return (
    candidate.type === TYPING_SEGMENT_CAPTURED_MESSAGE &&
    typeof candidate.segment === "object" &&
    candidate.segment !== null
  );
}

export function isCollectorContentStatusMessage(
  value: unknown,
): value is CollectorContentStatusMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<CollectorContentStatusMessage>;
  return (
    candidate.type === COLLECTOR_CONTENT_STATUS_MESSAGE &&
    typeof candidate.status === "object" &&
    candidate.status !== null
  );
}

export function isProactivePromptAvailableMessage(
  value: unknown,
): value is ProactivePromptAvailableMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ProactivePromptAvailableMessage>;
  return (
    candidate.type === PROACTIVE_PROMPT_AVAILABLE_MESSAGE &&
    (typeof candidate.conversationId === "string" ||
      candidate.conversationId === null) &&
    typeof candidate.promptText === "string" &&
    Array.isArray(candidate.suggestedPrompts)
  );
}

export function isProactivePromptDismissedMessage(
  value: unknown,
): value is ProactivePromptDismissedMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ProactivePromptDismissedMessage>;
  return candidate.type === PROACTIVE_PROMPT_DISMISSED_MESSAGE;
}

export function isChildSupportMessageRequestedMessage(
  value: unknown,
): value is ChildSupportMessageRequestedMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ChildSupportMessageRequestedMessage>;
  return (
    candidate.type === CHILD_SUPPORT_MESSAGE_REQUESTED_MESSAGE &&
    typeof candidate.message === "string"
  );
}
