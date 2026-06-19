export const DEFAULT_SERIES_PALETTE = [
  "#0072B2",
  "#D55E00",
  "#009E73",
  "#CC79A7",
  "#E69F00",
  "#56B4E9",
  "#F0E442",
  "#000000",
  "#7F3C8D",
  "#11A579",
  "#3969AC",
  "#F2B701",
];

export function seriesColors(series, overrides = {}) {
  const colors = new Map();
  const usedDefaults = new Set();
  for (const item of series) {
    const colorKey = item.colorKey ?? item.label;
    const override = overrides[colorKey];
    if (override) {
      colors.set(colorKey, override);
      continue;
    }
    const color = defaultColorForKey(colorKey, usedDefaults);
    usedDefaults.add(color);
    colors.set(colorKey, color);
  }
  return colors;
}

function defaultColorForKey(colorKey, usedDefaults) {
  const offset = stableIndex(colorKey, DEFAULT_SERIES_PALETTE.length);
  for (let index = 0; index < DEFAULT_SERIES_PALETTE.length; index += 1) {
    const color = DEFAULT_SERIES_PALETTE[(offset + index) % DEFAULT_SERIES_PALETTE.length];
    if (!usedDefaults.has(color)) return color;
  }
  return DEFAULT_SERIES_PALETTE[offset];
}

function stableIndex(value, modulo) {
  let hash = 0;
  for (const char of String(value)) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return hash % modulo;
}
