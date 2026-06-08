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
  return new Map(
    series.map((item, index) => {
      const colorKey = item.colorKey ?? item.label;
      return [
        item.label,
        overrides[colorKey] ??
          DEFAULT_SERIES_PALETTE[index % DEFAULT_SERIES_PALETTE.length],
      ];
    }),
  );
}
