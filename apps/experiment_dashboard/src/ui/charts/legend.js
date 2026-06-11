import { escapeHtml } from "../../shared/formatting/html.js";

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
