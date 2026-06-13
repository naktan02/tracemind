import type { TextDiff, TextDiffRange } from "./textDiff";
import { diffText, diffTextRange } from "./textDiff";

export type ImeEventContext = {
  now: Date;
  eventType?: string;
  inputType: string | null;
  insertedText: string | null;
  isCompositionUpdate: boolean;
};

export type ImeStateSnapshot = {
  lastText: string;
  compositionDraftText: string | null;
  lastCompositionText: string | null;
  lastCompositionAtMs: number | null;
};

const IME_PHANTOM_DELETE_GRACE_MS = 150;

export function isLikelyImePhantomDeletion(
  context: ImeEventContext,
  state: ImeStateSnapshot,
  diff: TextDiff,
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
  return state.lastCompositionText.endsWith(diff.deleted);
}

export function readCompositionCommitText(
  context: ImeEventContext,
  diff: TextDiff,
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

export function shouldKeepPendingComposition(
  context: ImeEventContext,
  state: ImeStateSnapshot,
  snapshotText: string,
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
  if (hasRemovableCompositionPlaceholder(snapshotText)) {
    return false;
  }
  return (
    endsWithKoreanCompositionPlaceholder(snapshotText) ||
    state.compositionDraftText !== null
  );
}

export function endsWithPendingKoreanComposition(text: string): boolean {
  return endsWithKoreanCompositionPlaceholder(text);
}

export function readObservedStableText(
  context: ImeEventContext,
  state: ImeStateSnapshot,
  snapshotText: string,
): string {
  if (!isCompositionCommit(context)) {
    return snapshotText;
  }
  return synthesizeCompositionEndText(
    state.lastText,
    state.compositionDraftText,
    snapshotText,
    context.insertedText,
  );
}

export function isCompositionCommit(context: ImeEventContext): boolean {
  return (
    context.eventType === "compositionend" ||
    (context.eventType === "input" &&
      !context.isCompositionUpdate &&
      (context.inputType ?? "").includes("Composition"))
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
  draftRange: Pick<TextDiffRange, "inserted" | "prefixLength" | "suffixLength">,
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
  if (
    prefixBeforePrevious === "" ||
    !committedText.startsWith(prefixBeforePrevious)
  ) {
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
  return mergeCommittedHangulSyllable(draftWithoutPlaceholder, committedText);
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
    if (sharesHangulLeadingSyllable(lastDraftChar, committedRemainderChars[0])) {
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
  return Array.from(text).some((char) => isKoreanCompositionPlaceholder(char));
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

function readCompatibilityConsonantIndex(
  char: string | undefined,
): number | null {
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
