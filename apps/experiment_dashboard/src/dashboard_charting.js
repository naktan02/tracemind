const DEFAULT_SERIES_PALETTE = [
  "#23766f",
  "#d97732",
  "#527a45",
  "#8f5b3d",
  "#6c7a89",
  "#a94f1f",
  "#5a6fb0",
  "#b55a7a",
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

export function renderSeriesLegend(series, colors, scope = null) {
  return `
    <div class="chart-legend editable-legend">
      ${series
        .map((item) => {
          const colorKey = item.colorKey ?? item.label;
          const color = colors.get(item.label);
          return `
            <span>
              <input
                class="legend-color-input"
                type="color"
                value="${escapeHtml(color)}"
                data-series-color-scope="${escapeHtml(scope ?? "")}"
                data-series-color-key="${escapeHtml(colorKey)}"
                aria-label="${escapeHtml(item.label)} 색상 변경"
              />
              ${escapeHtml(item.label)}
            </span>
          `;
        })
        .join("")}
    </div>
  `;
}

export function buildValueTicks(minValue, maxValue, tickCount) {
  if (tickCount <= 1) {
    return [minValue];
  }
  const step = (maxValue - minValue) / (tickCount - 1);
  return Array.from({ length: tickCount }, (_item, index) => minValue + step * index);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
