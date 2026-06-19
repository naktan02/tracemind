import type {
  TypingSegmentPayload,
  TypingSegmentStatsPayload,
  TypingSurfaceType,
} from "../contracts/generated";
import { normalizeEditorSnapshotText } from "./canonicalText";
import {
  endsWithPendingKoreanComposition,
  isCompositionCommit,
  isLikelyImePhantomDeletion,
  readCompositionCommitText,
  readObservedStableText,
  shouldKeepPendingComposition,
} from "./hangulIme";
import {
  chooseBetterSnapshotText,
  chooseFinalText,
} from "./segmentText";
import type { TextSurfaceSnapshot } from "./surfaceDetector";
import { diffText } from "./textDiff";

export type SegmentBuildContext = {
  elementId: string;
  snapshot: TextSurfaceSnapshot;
  now: Date;
  eventType?: string;
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
  baselineText: string;
  lastText: string;
  bestText: string;
  compositionDraftText: string | null;
  lastCompositionText: string | null;
  lastCompositionAtMs: number | null;
  pendingCompositionFlushDeferralCount: number;
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
  private readonly baselineTexts = new Map<string, string>();

  constructor(
    private readonly config: SegmentBufferConfig,
    private readonly emit: (segment: TypingSegmentPayload) => void,
  ) {}

  observe(context: SegmentBuildContext): void {
    const snapshotText = normalizeEditorSnapshotText(context.snapshot.text);
    const previous = this.states.get(context.elementId);
    const state =
      previous ??
      createInitialState(
        context,
        this.baselineTexts.get(context.elementId) ?? "",
      );
    updateStateMetadata(state, context);
    state.pendingCompositionFlushDeferralCount = 0;
    if (context.isCompositionUpdate) {
      state.compositionDraftText = snapshotText;
      state.stats.composition_count += 1;
      this.states.set(context.elementId, state);
      this.reschedule(context.elementId, state);
      return;
    }
    if (shouldKeepPendingComposition(context, state, snapshotText)) {
      state.compositionDraftText = snapshotText;
      state.stats.composition_count += 1;
      this.states.set(context.elementId, state);
      this.reschedule(context.elementId, state);
      return;
    }

    const observedText = readObservedStableText(context, state, snapshotText);
    const diff = diffText(state.lastText, observedText);
    const shouldTrackTextDiff =
      context.eventType !== "beforeinput" && !context.isCompositionUpdate;
    const isDeletionEvent = (context.inputType ?? "").startsWith("delete");
    const shouldSuppressImeDeletion = isLikelyImePhantomDeletion(
      context,
      state,
      diff,
      isDeletionEvent,
    );
    const compositionCommitText = readCompositionCommitText(context, diff);

    state.lastText = shouldSuppressImeDeletion
      ? state.lastText
      : observedText;
    if (shouldTrackTextDiff && isDeletionEvent && !shouldSuppressImeDeletion) {
      state.bestText = state.lastText;
    } else if (shouldTrackTextDiff) {
      state.bestText = chooseBetterSnapshotText(
        state.baselineText,
        state.bestText,
        state.lastText,
      );
    }
    state.stats.insert_count +=
      shouldTrackTextDiff && diff.inserted.length > 0 ? 1 : 0;
    state.stats.delete_count +=
      shouldTrackTextDiff &&
      isDeletionEvent &&
      !shouldSuppressImeDeletion &&
      diff.deleted.length > 0
        ? 1
        : 0;
    state.stats.paste_count += context.inputType === "insertFromPaste" ? 1 : 0;
    state.stats.composition_count += context.isCompositionUpdate ? 1 : 0;
    if (
      shouldTrackTextDiff &&
      isDeletionEvent &&
      !shouldSuppressImeDeletion &&
      diff.deleted.length > 0
    ) {
      state.deletedTextParts.push(diff.deleted);
    }
    if (
      isCompositionCommit(context) &&
      compositionCommitText !== null &&
      compositionCommitText !== ""
    ) {
      state.lastCompositionText = compositionCommitText;
      state.lastCompositionAtMs = context.now.getTime();
    }
    state.compositionDraftText = null;

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
    if (state.timerId !== null) {
      window.clearTimeout(state.timerId);
      state.timerId = null;
    }
    if (state.compositionDraftText !== null) {
      this.reschedule(elementId, state);
      return;
    }
    if (shouldDeferPendingCompositionFlush(state)) {
      state.pendingCompositionFlushDeferralCount += 1;
      this.reschedule(elementId, state);
      return;
    }
    this.states.delete(elementId);
    const finalText = readFinalText(state);
    const deletedText = state.deletedTextParts.join(" ").trim();
    this.baselineTexts.set(elementId, state.lastText);
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

function readFinalText(state: SegmentState): string {
  if (state.surfaceType === "input") {
    return state.lastText.trim();
  }
  return chooseFinalText(
    state.baselineText.trim(),
    state.lastText.trim(),
    state.bestText.trim(),
  );
}

function createInitialState(
  context: SegmentBuildContext,
  baselineText: string,
): SegmentState {
  return {
    elementId: context.elementId,
    startedAt: context.now,
    lastUpdatedAt: context.now,
    baselineText,
    lastText: baselineText,
    bestText: baselineText,
    compositionDraftText: null,
    lastCompositionText: null,
    lastCompositionAtMs: null,
    pendingCompositionFlushDeferralCount: 0,
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

function shouldDeferPendingCompositionFlush(state: SegmentState): boolean {
  const finalText = readFinalText(state);
  return (
    state.pendingCompositionFlushDeferralCount < 2 &&
    state.surfaceType !== "input" &&
    (endsWithPendingKoreanComposition(state.lastText) ||
      endsWithPendingKoreanComposition(state.bestText) ||
      endsWithPendingKoreanComposition(finalText))
  );
}

function updateStateMetadata(
  state: SegmentState,
  context: SegmentBuildContext,
): void {
  state.lastUpdatedAt = context.now;
  state.surfaceType = context.snapshot.surfaceType;
  state.captureConfidence = context.snapshot.captureConfidence;
  state.fieldHint = context.snapshot.fieldHint;
  state.locale = context.locale ?? "ko";
}

function buildSegmentId(elementId: string, endedAt: Date): string {
  const safeElementId = elementId.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 48);
  const randomPart = Math.random().toString(36).slice(2, 10);
  return `typing_${safeElementId}_${endedAt.getTime()}_${randomPart}`;
}
