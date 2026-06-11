import { diffText } from "./textDiff";

export function chooseFinalText(
  baselineText: string,
  snapshotText: string,
  bestText: string,
): string {
  const snapshotCandidate = readSnapshotCandidate(baselineText, snapshotText);
  const bestCandidate = readSnapshotCandidate(baselineText, bestText);
  return bestCandidate.length >= snapshotCandidate.length
    ? bestCandidate
    : snapshotCandidate;
}

export function chooseBetterSnapshotText(
  baselineText: string,
  currentBestText: string,
  nextText: string,
): string {
  const currentCandidate = readSnapshotCandidate(baselineText, currentBestText);
  const nextCandidate = readSnapshotCandidate(baselineText, nextText);
  return nextCandidate.length >= currentCandidate.length
    ? nextText
    : currentBestText;
}

function readSnapshotCandidate(
  baselineText: string,
  snapshotText: string,
): string {
  if (!baselineText) {
    return snapshotText;
  }
  const diff = diffText(baselineText, snapshotText);
  if (diff.inserted) {
    return diff.inserted.trim();
  }
  return "";
}
