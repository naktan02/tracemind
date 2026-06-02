import type { TypingSegmentPayload } from "../contracts/generated";

export const TYPING_SEGMENT_CAPTURED_MESSAGE = "tracemind.typingSegmentCaptured";

export type TypingSegmentCapturedMessage = {
  type: typeof TYPING_SEGMENT_CAPTURED_MESSAGE;
  segment: TypingSegmentPayload;
};

export type ExtensionMessage = TypingSegmentCapturedMessage;

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
