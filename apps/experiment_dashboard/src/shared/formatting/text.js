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

export function compactDateTime(value) {
  const text = String(value ?? "").trim();
  if (!text) return "-";
  const match = text.match(
    /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/,
  );
  if (!match) return text.slice(0, 16);
  return `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}`;
}

export function compactDate(value) {
  const text = compactDateTime(value);
  return text === "-" ? text : text.slice(0, 10);
}
