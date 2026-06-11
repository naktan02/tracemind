export function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatMetric(value) {
  const number = numberOrNull(value);
  if (number === null) {
    return "-";
  }
  if (Math.abs(number) >= 1000) {
    return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export function formatCount(value) {
  const number = numberOrNull(value);
  return number === null ? "-" : number.toLocaleString();
}

export function formatBytes(value) {
  const number = numberOrNull(value);
  if (number === null) return "-";
  if (number >= 1024 * 1024) return `${formatMetric(number / 1024 / 1024)} MB`;
  if (number >= 1024) return `${formatMetric(number / 1024)} KB`;
  return `${formatCount(number)} B`;
}

export function formatSeconds(value) {
  const number = numberOrNull(value);
  return number === null ? "-" : `${formatMetric(number)}s`;
}
