import type {
  TypingSegmentPayload,
  TypingSegmentStatsPayload,
  TypingSurfaceType,
} from "../contracts/generated";
import type { TextSurfaceSnapshot } from "./surfaceDetector";

export type SegmentBuildContext = {
  elementId: string;
  snapshot: TextSurfaceSnapshot;
  now: Date;
  inputType: string | null;
  insertedText: string | null;
  isCompositionUpdate: boolean;
  locale?: string;
};

export type SegmentBufferConfig = {
  idleMs: number;
  sourceType: TypingSegmentPayload["source_type"];
};

type SegmentState = {
  elementId: string;
  startedAt: Date;
  lastUpdatedAt: Date;
  lastText: string;
  deletedTextParts: string[];
  surfaceType: TypingSurfaceType;
  captureConfidence: TypingSegmentPayload["capture_confidence"];
  fieldHint: string | null;
  locale: string;
  stats: TypingSegmentStatsPayload;
  timerId: number | null;
};

export class SegmentBuffer {
  private readonly states = new Map<string, SegmentState>();

  constructor(
    private readonly config: SegmentBufferConfig,
    private readonly emit: (segment: TypingSegmentPayload) => void,
  ) {}

  observe(context: SegmentBuildContext): void {
    const previous = this.states.get(context.elementId);
    const state = previous ?? createInitialState(context);
    const diff = diffText(state.lastText, context.snapshot.text);
    const shouldTrackTextDiff = !context.isCompositionUpdate;

    state.lastText = context.snapshot.text;
    state.lastUpdatedAt = context.now;
    state.surfaceType = context.snapshot.surfaceType;
    state.captureConfidence = context.snapshot.captureConfidence;
    state.fieldHint = context.snapshot.fieldHint;
    state.locale = context.locale ?? "ko";
    state.stats.insert_count +=
      shouldTrackTextDiff && diff.inserted.length > 0 ? 1 : 0;
    state.stats.delete_count +=
      shouldTrackTextDiff && diff.deleted.length > 0 ? 1 : 0;
    state.stats.paste_count += context.inputType === "insertFromPaste" ? 1 : 0;
    state.stats.composition_count += context.isCompositionUpdate ? 1 : 0;
    if (shouldTrackTextDiff && diff.deleted.length > 0) {
      state.deletedTextParts.push(diff.deleted);
    }

    this.states.set(context.elementId, state);
    this.reschedule(context.elementId, state);
  }

  flushElement(elementId: string): void {
    this.flush(elementId);
  }

  flushAll(): void {
    for (const elementId of Array.from(this.states.keys())) {
      this.flush(elementId);
    }
  }

  private reschedule(elementId: string, state: SegmentState): void {
    if (state.timerId !== null) {
      window.clearTimeout(state.timerId);
    }
    state.timerId = window.setTimeout(() => {
      this.flush(elementId);
    }, this.config.idleMs);
  }

  private flush(elementId: string): void {
    const state = this.states.get(elementId);
    if (state === undefined) {
      return;
    }
    this.states.delete(elementId);
    if (state.timerId !== null) {
      window.clearTimeout(state.timerId);
    }
    const finalText = state.lastText.trim();
    const deletedText = state.deletedTextParts.join(" ").trim();
    if (!finalText && !deletedText) {
      return;
    }

    this.emit({
      schema_version: "typing_segment.v1",
      segment_id: buildSegmentId(elementId, state.lastUpdatedAt),
      source_type: this.config.sourceType,
      surface_type: state.surfaceType,
      capture_confidence: state.captureConfidence,
      page_origin: window.location.origin,
      page_url: window.location.href,
      field_hint: state.fieldHint,
      started_at: state.startedAt.toISOString(),
      ended_at: state.lastUpdatedAt.toISOString(),
      idle_ms: this.config.idleMs,
      locale: state.locale,
      final_text: finalText || null,
      deleted_text: deletedText || null,
      stats: state.stats,
    });
  }
}

function createInitialState(context: SegmentBuildContext): SegmentState {
  return {
    elementId: context.elementId,
    startedAt: context.now,
    lastUpdatedAt: context.now,
    lastText: "",
    deletedTextParts: [],
    surfaceType: context.snapshot.surfaceType,
    captureConfidence: context.snapshot.captureConfidence,
    fieldHint: context.snapshot.fieldHint,
    locale: context.locale ?? "ko",
    stats: {
      insert_count: 0,
      delete_count: 0,
      paste_count: 0,
      composition_count: 0,
    },
    timerId: null,
  };
}

function diffText(
  previous: string,
  current: string,
): { inserted: string; deleted: string } {
  if (previous === current) {
    return { inserted: "", deleted: "" };
  }

  let prefixLength = 0;
  const maxPrefixLength = Math.min(previous.length, current.length);
  while (
    prefixLength < maxPrefixLength &&
    previous[prefixLength] === current[prefixLength]
  ) {
    prefixLength += 1;
  }

  let suffixLength = 0;
  while (
    suffixLength < previous.length - prefixLength &&
    suffixLength < current.length - prefixLength &&
    previous[previous.length - 1 - suffixLength] ===
      current[current.length - 1 - suffixLength]
  ) {
    suffixLength += 1;
  }

  return {
    deleted: previous.slice(prefixLength, previous.length - suffixLength),
    inserted: current.slice(prefixLength, current.length - suffixLength),
  };
}

function buildSegmentId(elementId: string, endedAt: Date): string {
  const safeElementId = elementId.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 48);
  const randomPart = Math.random().toString(36).slice(2, 10);
  return `typing_${safeElementId}_${endedAt.getTime()}_${randomPart}`;
}
