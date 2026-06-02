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
  typedText: string;
  lastCompositionText: string | null;
  lastCompositionAtMs: number | null;
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
    const previous = this.states.get(context.elementId);
    const state =
      previous ??
      createInitialState(
        context,
        this.baselineTexts.get(context.elementId) ?? "",
      );
    updateStateMetadata(state, context);
    if (context.isCompositionUpdate) {
      state.compositionDraftText = context.snapshot.text;
      state.stats.composition_count += 1;
      this.states.set(context.elementId, state);
      this.reschedule(context.elementId, state);
      return;
    }
    if (shouldKeepPendingComposition(context, state)) {
      state.compositionDraftText = context.snapshot.text;
      state.stats.composition_count += 1;
      this.states.set(context.elementId, state);
      this.reschedule(context.elementId, state);
      return;
    }

    const observedText = readObservedStableText(context, state);
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
    const committedInsertedText = readCommittedInsertedText(context, diff);

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
      state.typedText = removeDeletedSuffix(state.typedText, diff.deleted);
    }
    if (
      isCompositionCommit(context) &&
      diff.deleted.length > 0 &&
      !shouldSuppressImeDeletion
    ) {
      state.typedText = removeDeletedSuffix(state.typedText, diff.deleted);
    }
    if (committedInsertedText !== null && committedInsertedText !== "") {
      state.typedText = `${state.typedText}${committedInsertedText}`;
      if (isCompositionCommit(context)) {
        state.lastCompositionText = committedInsertedText;
        state.lastCompositionAtMs = context.now.getTime();
      }
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
    this.states.delete(elementId);
    const finalText = chooseFinalText(
      state.baselineText.trim(),
      state.lastText.trim(),
      state.bestText.trim(),
      state.typedText.trim(),
    );
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
    typedText: "",
    lastCompositionText: null,
    lastCompositionAtMs: null,
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

const IME_PHANTOM_DELETE_GRACE_MS = 150;

function isLikelyImePhantomDeletion(
  context: SegmentBuildContext,
  state: SegmentState,
  diff: { inserted: string; deleted: string },
  isDeletionEvent: boolean,
): boolean {
  if (
    !isDeletionEvent ||
    context.eventType !== "input" ||
    context.isCompositionUpdate ||
    diff.deleted === "" ||
    diff.inserted !== "" ||
    state.lastCompositionText === null ||
    state.lastCompositionAtMs === null
  ) {
    return false;
  }
  if (
    context.now.getTime() - state.lastCompositionAtMs >
    IME_PHANTOM_DELETE_GRACE_MS
  ) {
    return false;
  }
  return (
    state.lastText.trim() === state.typedText.trim() &&
    state.lastCompositionText.endsWith(diff.deleted)
  );
}

function readCommittedInsertedText(
  context: SegmentBuildContext,
  diff: { inserted: string; deleted: string },
): string | null {
  if (context.eventType === "compositionend") {
    return diff.inserted || context.insertedText;
  }
  if (context.eventType !== "input" || context.isCompositionUpdate) {
    return null;
  }
  if (context.inputType === "insertFromPaste") {
    return context.insertedText ?? diff.inserted;
  }
  if ((context.inputType ?? "").startsWith("insert")) {
    return context.insertedText ?? diff.inserted;
  }
  return null;
}

function shouldKeepPendingComposition(
  context: SegmentBuildContext,
  state: SegmentState,
): boolean {
  if (
    context.eventType !== "input" ||
    context.isCompositionUpdate ||
    !(context.inputType ?? "").includes("Composition")
  ) {
    return false;
  }
  const hasUnreliableCommittedText =
    context.insertedText === null ||
    context.insertedText === "" ||
    containsOnlyKoreanCompositionPlaceholders(context.insertedText);
  if (!hasUnreliableCommittedText) {
    return false;
  }
  if (hasRemovableCompositionPlaceholder(context.snapshot.text)) {
    return false;
  }
  return (
    endsWithKoreanCompositionPlaceholder(context.snapshot.text) ||
    state.compositionDraftText !== null
  );
}

function readObservedStableText(
  context: SegmentBuildContext,
  state: SegmentState,
): string {
  if (!isCompositionCommit(context)) {
    return context.snapshot.text;
  }
  return synthesizeCompositionEndText(
    state.lastText,
    state.compositionDraftText,
    context.snapshot.text,
    context.insertedText,
  );
}

function synthesizeCompositionEndText(
  stableText: string,
  draftText: string | null,
  snapshotText: string,
  committedText: string | null,
): string {
  if (draftText === null || committedText === null || committedText === "") {
    return normalizeCommittedCompositionSnapshot(snapshotText);
  }
  const committedDocumentCandidate = readCommittedDocumentCandidate(
    stableText,
    snapshotText,
    committedText,
  );
  if (committedDocumentCandidate !== null) {
    return committedDocumentCandidate;
  }
  const snapshotDiff = diffText(stableText, snapshotText);
  if (
    snapshotDiff.inserted !== "" &&
    !containsKoreanCompositionPlaceholder(snapshotDiff.inserted) &&
    snapshotText !== draftText
  ) {
    return snapshotText;
  }
  const draftRange = diffTextRange(stableText, draftText);
  if (draftRange.inserted === "") {
    return snapshotText;
  }
  const stableOverlapText = readStableOverlapCompositionText(
    draftText,
    draftRange,
    committedText,
  );
  if (stableOverlapText !== null) {
    return stableOverlapText;
  }
  const committedInsertion = chooseCommittedInsertion(
    draftRange.inserted,
    committedText,
  );
  return [
    draftText.slice(0, draftRange.prefixLength),
    committedInsertion,
    draftRange.suffixLength === 0
      ? ""
      : draftText.slice(draftText.length - draftRange.suffixLength),
  ].join("");
}

function readCommittedDocumentCandidate(
  stableText: string,
  snapshotText: string,
  committedText: string,
): string | null {
  if (
    committedText.length <= snapshotText.length ||
    containsKoreanCompositionPlaceholder(committedText)
  ) {
    return null;
  }
  if (stableText === "" || committedText.startsWith(stableText)) {
    return committedText;
  }
  return null;
}

function normalizeCommittedCompositionSnapshot(snapshotText: string): string {
  const chars = Array.from(snapshotText);
  const normalizedChars: string[] = [];
  for (let index = 0; index < chars.length; index += 1) {
    const currentChar = chars[index];
    const nextChar = chars[index + 1];
    if (
      isKoreanCompositionPlaceholder(currentChar) &&
      startsWithSameKoreanConsonant(currentChar, nextChar)
    ) {
      continue;
    }
    normalizedChars.push(currentChar);
  }
  return normalizedChars.join("");
}

function hasRemovableCompositionPlaceholder(snapshotText: string): boolean {
  return normalizeCommittedCompositionSnapshot(snapshotText) !== snapshotText;
}

function readStableOverlapCompositionText(
  draftText: string,
  draftRange: {
    inserted: string;
    prefixLength: number;
    suffixLength: number;
  },
  committedText: string,
): string | null {
  if (
    draftRange.prefixLength === 0 ||
    !containsOnlyKoreanCompositionPlaceholders(draftRange.inserted)
  ) {
    return null;
  }
  const previousStableChar = draftText[draftRange.prefixLength - 1];
  const committedChars = Array.from(committedText);
  const prefixBeforePrevious = draftText.slice(0, draftRange.prefixLength - 1);
  const suffixText =
    draftRange.suffixLength === 0
      ? ""
      : draftText.slice(draftText.length - draftRange.suffixLength);
  const committedDocumentText = readCommittedDocumentText(
    prefixBeforePrevious,
    previousStableChar,
    suffixText,
    committedText,
  );
  if (committedDocumentText !== null) {
    return committedDocumentText;
  }
  if (
    committedChars.length !== 1 ||
    !sharesHangulLeadingConsonant(previousStableChar, committedChars[0])
  ) {
    return null;
  }
  return [
    draftText.slice(0, draftRange.prefixLength - 1),
    committedText,
    draftRange.suffixLength === 0
      ? ""
      : draftText.slice(draftText.length - draftRange.suffixLength),
  ].join("");
}

function readCommittedDocumentText(
  prefixBeforePrevious: string,
  previousStableChar: string | undefined,
  suffixText: string,
  committedText: string,
): string | null {
  if (prefixBeforePrevious === "" || !committedText.startsWith(prefixBeforePrevious)) {
    return null;
  }
  const remainderChars = Array.from(
    committedText.slice(prefixBeforePrevious.length),
  );
  if (!sharesHangulLeadingConsonant(previousStableChar, remainderChars[0])) {
    return null;
  }
  if (suffixText === "" || committedText.endsWith(suffixText)) {
    return committedText;
  }
  return `${committedText}${suffixText}`;
}

function chooseCommittedInsertion(
  draftInsertedText: string,
  committedText: string,
): string {
  if (draftInsertedText === committedText) {
    return committedText;
  }
  const draftWithoutPlaceholder = removeTrailingKoreanCompositionPlaceholders(
    draftInsertedText,
  );
  if (draftWithoutPlaceholder === draftInsertedText) {
    return committedText;
  }
  return mergeCommittedHangulSyllable(
    draftWithoutPlaceholder,
    committedText,
  );
}

function mergeCommittedHangulSyllable(
  draftPrefixText: string,
  committedText: string,
): string {
  if (draftPrefixText !== "" && committedText.startsWith(draftPrefixText)) {
    return committedText;
  }
  const draftPrefixChars = Array.from(draftPrefixText);
  const committedChars = Array.from(committedText);
  const lastDraftChar = draftPrefixChars[draftPrefixChars.length - 1];
  const firstCommittedChar = committedChars[0];
  if (
    draftPrefixChars.length === 1 &&
    committedChars.length === 1 &&
    sharesHangulLeadingConsonant(lastDraftChar, firstCommittedChar)
  ) {
    return committedText;
  }
  const prefixBeforeLastDraft = draftPrefixChars.slice(0, -1).join("");
  if (
    prefixBeforeLastDraft !== "" &&
    committedText.startsWith(prefixBeforeLastDraft)
  ) {
    const committedRemainderChars = Array.from(
      committedText.slice(prefixBeforeLastDraft.length),
    );
    if (
      sharesHangulLeadingSyllable(
        lastDraftChar,
        committedRemainderChars[0],
      )
    ) {
      return committedText;
    }
  }
  if (
    committedChars.length === 1 &&
    sharesHangulLeadingSyllable(lastDraftChar, firstCommittedChar)
  ) {
    return `${draftPrefixChars.slice(0, -1).join("")}${committedText}`;
  }
  return `${draftPrefixText}${committedText}`;
}

function removeTrailingKoreanCompositionPlaceholders(text: string): string {
  const chars = Array.from(text);
  while (
    chars.length > 0 &&
    isKoreanCompositionPlaceholder(chars[chars.length - 1])
  ) {
    chars.pop();
  }
  return chars.join("");
}

function containsOnlyKoreanCompositionPlaceholders(text: string): boolean {
  const chars = Array.from(text);
  return (
    chars.length > 0 &&
    chars.every((char) => isKoreanCompositionPlaceholder(char))
  );
}

function containsKoreanCompositionPlaceholder(text: string): boolean {
  return Array.from(text).some((char) =>
    isKoreanCompositionPlaceholder(char),
  );
}

function endsWithKoreanCompositionPlaceholder(text: string): boolean {
  const chars = Array.from(text);
  return (
    chars.length > 0 &&
    isKoreanCompositionPlaceholder(chars[chars.length - 1])
  );
}

function isKoreanCompositionPlaceholder(char: string | undefined): boolean {
  if (char === undefined) {
    return false;
  }
  const codePoint = char.codePointAt(0);
  if (codePoint === undefined) {
    return false;
  }
  return (
    (codePoint >= 0x1100 && codePoint <= 0x11ff) ||
    (codePoint >= 0x3130 && codePoint <= 0x318f) ||
    (codePoint >= 0xa960 && codePoint <= 0xa97f) ||
    (codePoint >= 0xd7b0 && codePoint <= 0xd7ff)
  );
}

function sharesHangulLeadingSyllable(
  draftChar: string | undefined,
  committedChar: string | undefined,
): boolean {
  const draftParts = decomposeHangulSyllable(draftChar);
  const committedParts = decomposeHangulSyllable(committedChar);
  return (
    draftParts !== null &&
    committedParts !== null &&
    draftParts.leadingConsonantIndex === committedParts.leadingConsonantIndex &&
    draftParts.vowelIndex === committedParts.vowelIndex
  );
}

function sharesHangulLeadingConsonant(
  draftChar: string | undefined,
  committedChar: string | undefined,
): boolean {
  const draftParts = decomposeHangulSyllable(draftChar);
  const committedParts = decomposeHangulSyllable(committedChar);
  return (
    draftParts !== null &&
    committedParts !== null &&
    draftParts.leadingConsonantIndex === committedParts.leadingConsonantIndex
  );
}

function startsWithSameKoreanConsonant(
  placeholderChar: string | undefined,
  committedChar: string | undefined,
): boolean {
  const placeholderIndex = readCompatibilityConsonantIndex(placeholderChar);
  const committedParts = decomposeHangulSyllable(committedChar);
  return (
    placeholderIndex !== null &&
    committedParts !== null &&
    placeholderIndex === committedParts.leadingConsonantIndex
  );
}

function readCompatibilityConsonantIndex(char: string | undefined): number | null {
  switch (char) {
    case "ㄱ":
      return 0;
    case "ㄲ":
      return 1;
    case "ㄴ":
      return 2;
    case "ㄷ":
      return 3;
    case "ㄸ":
      return 4;
    case "ㄹ":
      return 5;
    case "ㅁ":
      return 6;
    case "ㅂ":
      return 7;
    case "ㅃ":
      return 8;
    case "ㅅ":
      return 9;
    case "ㅆ":
      return 10;
    case "ㅇ":
      return 11;
    case "ㅈ":
      return 12;
    case "ㅉ":
      return 13;
    case "ㅊ":
      return 14;
    case "ㅋ":
      return 15;
    case "ㅌ":
      return 16;
    case "ㅍ":
      return 17;
    case "ㅎ":
      return 18;
    default:
      return null;
  }
}

function decomposeHangulSyllable(char: string | undefined): {
  leadingConsonantIndex: number;
  vowelIndex: number;
} | null {
  if (char === undefined) {
    return null;
  }
  const codePoint = char.codePointAt(0);
  if (
    codePoint === undefined ||
    codePoint < HANGUL_SYLLABLE_START ||
    codePoint > HANGUL_SYLLABLE_END
  ) {
    return null;
  }
  const syllableIndex = codePoint - HANGUL_SYLLABLE_START;
  return {
    leadingConsonantIndex: Math.floor(syllableIndex / HANGUL_VOWEL_BLOCK_SIZE),
    vowelIndex: Math.floor(
      (syllableIndex % HANGUL_VOWEL_BLOCK_SIZE) / HANGUL_TRAILING_COUNT,
    ),
  };
}

const HANGUL_SYLLABLE_START = 0xac00;
const HANGUL_SYLLABLE_END = 0xd7a3;
const HANGUL_TRAILING_COUNT = 28;
const HANGUL_VOWEL_BLOCK_SIZE = 21 * HANGUL_TRAILING_COUNT;

function isCompositionCommit(context: SegmentBuildContext): boolean {
  return (
    context.eventType === "compositionend" ||
    (context.eventType === "input" &&
      !context.isCompositionUpdate &&
      (context.inputType ?? "").includes("Composition"))
  );
}

function chooseFinalText(
  baselineText: string,
  snapshotText: string,
  bestText: string,
  typedText: string,
): string {
  const snapshotCandidate = readSnapshotCandidate(baselineText, snapshotText);
  const bestCandidate = readSnapshotCandidate(baselineText, bestText);
  const bestObservedCandidate =
    bestCandidate.length >= snapshotCandidate.length
      ? bestCandidate
      : snapshotCandidate;
  if (bestObservedCandidate && bestObservedCandidate.length > typedText.length) {
    return bestObservedCandidate;
  }
  if (typedText) {
    return typedText;
  }
  return bestObservedCandidate;
}

function readSnapshotCandidate(baselineText: string, snapshotText: string): string {
  if (!baselineText) {
    return snapshotText;
  }
  const diff = diffText(baselineText, snapshotText);
  if (diff.inserted) {
    return diff.inserted.trim();
  }
  return "";
}

function chooseBetterSnapshotText(
  baselineText: string,
  currentBestText: string,
  nextText: string,
): string {
  const currentCandidate = readSnapshotCandidate(baselineText, currentBestText);
  const nextCandidate = readSnapshotCandidate(baselineText, nextText);
  return nextCandidate.length >= currentCandidate.length ? nextText : currentBestText;
}

function removeDeletedSuffix(text: string, deletedText: string): string {
  if (deletedText !== "" && text.endsWith(deletedText)) {
    return text.slice(0, text.length - deletedText.length);
  }
  return Array.from(text).slice(0, -1).join("");
}

function diffText(
  previous: string,
  current: string,
): { inserted: string; deleted: string } {
  const diff = diffTextRange(previous, current);
  return { inserted: diff.inserted, deleted: diff.deleted };
}

function diffTextRange(
  previous: string,
  current: string,
): {
  inserted: string;
  deleted: string;
  prefixLength: number;
  suffixLength: number;
} {
  if (previous === current) {
    return { inserted: "", deleted: "", prefixLength: 0, suffixLength: 0 };
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
    prefixLength,
    suffixLength,
  };
}

function buildSegmentId(elementId: string, endedAt: Date): string {
  const safeElementId = elementId.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 48);
  const randomPart = Math.random().toString(36).slice(2, 10);
  return `typing_${safeElementId}_${endedAt.getTime()}_${randomPart}`;
}
