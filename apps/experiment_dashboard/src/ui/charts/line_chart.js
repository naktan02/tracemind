import { escapeHtml } from "../../shared/formatting/html.js";
import { formatMetric } from "../../shared/formatting/numbers.js";
import { metricLabel } from "../../shared/formatting/metrics.js";
import { renderSeriesLegend } from "./legend.js";
import { seriesColors } from "./palette.js";

export function drawLineChart({
  series,
  metric,
  scope,
  xKey,
  xLabel,
  xTickFormatter,
  colorOverrides,
  axisLabel,
  storedAxisLabel = "",
  subtitle = "",
  width = 1120,
  height = 500,
}) {
  const allPoints = series.flatMap((item) => item.points);
  if (series.length === 0 || allPoints.length === 0) {
    return `<p class="empty">선택한 지표 값이 없습니다.</p>`;
  }

  const pad = { top: 42, right: 52, bottom: 72, left: 140 };
  const chartHeight = height - pad.top - pad.bottom;
  const pointInset = 40;
  const chartWidth = width - pad.left - pad.right - pointInset * 2;
  const xValues = allPoints.map((point) => point[xKey]);
  const yValues = allPoints.map((point) => point.value);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const xRange = Math.max(maxX - minX, 1);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const yPadding = Math.max((maxY - minY) * 0.08, 0.02);
  let axisMin = minY - yPadding;
  let axisMax = maxY + yPadding;
  if (minY >= 0 && maxY <= 1) {
    axisMin = Math.max(0, axisMin);
    axisMax = Math.min(1, axisMax);
  }
  const yRange = Math.max(axisMax - axisMin, 0.000001);
  const colors = seriesColors(series, colorOverrides);
  const xForPoint = (point) =>
    pad.left + pointInset + ((point[xKey] - minX) / xRange) * chartWidth;
  const yForValue = (value) =>
    pad.top + chartHeight - ((value - axisMin) / yRange) * chartHeight;
  const axisTitle = axisLabel || metricLabel(metric);

  const lines = series
    .map((item) => {
      const colorKey = item.colorKey ?? item.label;
      const color = colors.get(colorKey);
      const path = item.points
        .map((point) => `${xForPoint(point)},${yForValue(point.value)}`)
        .join(" ");
      const dots = item.points
        .map(
          (point) => `
            <circle cx="${xForPoint(point)}" cy="${yForValue(point.value)}" r="4" style="--series-color:${color}" data-series-color-key="${escapeHtml(colorKey)}">
              <title>${escapeHtml(item.label)} · ${escapeHtml(xLabel(point))} · ${formatMetric(point.value)}</title>
            </circle>
          `,
        )
        .join("");
      return `<polyline points="${path}" fill="none" style="--series-color:${color}" data-series-color-key="${escapeHtml(colorKey)}" />${dots}`;
    })
    .join("");

  const xTicks = uniqueSorted(xValues)
    .filter((_value, index, values) => {
      return (
        index === 0 ||
        index === values.length - 1 ||
        values.length <= 12 ||
        index % Math.ceil(values.length / 10) === 0
      );
    })
    .map((value) => {
      const x = xForPoint({ [xKey]: value });
      return `<text class="axis-label" x="${x}" y="${height - 24}" text-anchor="middle">${escapeHtml(xTickFormatter(value))}</text>`;
    })
    .join("");

  const yTicks = buildValueTicks(axisMin, axisMax, 5)
    .map((value) => {
      const y = yForValue(value);
      return `
        <line class="grid-line" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" />
        <text class="axis-label" x="${pad.left - 10}" y="${y + 4}" text-anchor="end">${formatMetric(value)}</text>
      `;
    })
    .join("");

  return `
    ${subtitle ? `<p class="chart-subtitle">${escapeHtml(subtitle)}</p>` : ""}
    <label class="chart-axis-label-control">
      Y Axis Label
      <input
        type="text"
        data-chart-axis-label-scope="${escapeHtml(scope)}"
        value="${escapeHtml(storedAxisLabel)}"
        placeholder="${escapeHtml(metricLabel(metric))}"
      />
    </label>
    ${renderSeriesLegend(series, colors, scope)}
    <div class="chart-scroll line-chart">
      <svg viewBox="0 0 ${width} ${height}" role="img">
        <text
          class="axis-title"
          data-chart-axis-label="${escapeHtml(scope)}"
          x="28"
          y="${pad.top + chartHeight / 2}"
          text-anchor="middle"
          transform="rotate(-90 28 ${pad.top + chartHeight / 2})"
        >${escapeHtml(axisTitle)}</text>
        <line class="axis-line" x1="${pad.left}" y1="${pad.top + chartHeight}" x2="${width - pad.right}" y2="${pad.top + chartHeight}" />
        <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + chartHeight}" />
        ${yTicks}
        ${lines}
        ${xTicks}
      </svg>
    </div>
  `;
}

function buildValueTicks(minValue, maxValue, tickCount) {
  if (tickCount <= 1) return [minValue];
  const step = (maxValue - minValue) / (tickCount - 1);
  return Array.from({ length: tickCount }, (_item, index) => minValue + step * index);
}

function uniqueSorted(values) {
  return Array.from(new Set(values)).sort((left, right) => left - right);
}
