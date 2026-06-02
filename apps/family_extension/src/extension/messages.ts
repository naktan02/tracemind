import type { TypingSegmentPayload } from "../contracts/generated";

export const TYPING_SEGMENT_CAPTURED_MESSAGE = "tracemind.typingSegmentCaptured";
export const COLLECTOR_CONTENT_STATUS_MESSAGE =
  "tracemind.collectorContentStatus";

export type TypingSegmentCapturedMessage = {
  type: typeof TYPING_SEGMENT_CAPTURED_MESSAGE;
  segment: TypingSegmentPayload;
};

export type CollectorContentStatusMessage = {
  type: typeof COLLECTOR_CONTENT_STATUS_MESSAGE;
  status: Record<string, unknown>;
};

export type ExtensionMessage =
  | TypingSegmentCapturedMessage
  | CollectorContentStatusMessage;

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
