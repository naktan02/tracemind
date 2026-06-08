export function uniqueValues(values) {
  return Array.from(new Set(values.filter((value) => value !== null && value !== undefined)));
}

export function shortRun(runId) {
  const text = String(runId ?? "-");
  return text.length <= 34 ? text : `${text.slice(0, 16)}...${text.slice(-12)}`;
}

export function shortSplit(selectionSlug) {
  return String(selectionSlug ?? "-")
    .replace(/labeled-/g, "L:")
    .replace(/unlabeled-/g, "U:")
    .replace(/validation-/g, "V:")
    .replace(/test-/g, "T:")
    .replace(/_/g, " ");
}
