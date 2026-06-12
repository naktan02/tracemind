import { escapeHtml } from "../../../shared/formatting/html.js";
import { metricLabel } from "../../../shared/formatting/metrics.js";
import { formatMetric, numberOrNull } from "../../../shared/formatting/numbers.js";
import { drawLineChart } from "../../../ui/charts/line_chart.js";
import { renderSelectedRunCard } from "../../../ui/controls/selected_run_card.js";
import {
  algorithmName,
  compareDisplayLabel,
  runDescriptor,
  runDetail,
} from "../logic/labels.js";
import {
  centralEpochMetricKeys,
  centralEpochPoints,
  centralLatestMetricValue,
  formatStepTick,
} from "../logic/metrics.js";
import { centralInitialPoint } from "../logic/initial_eval.js";

export function normalizeCompareSelection(rows, state, bundle) {
  const metrics = centralEpochMetricKeys(bundle);
  if (!metrics.includes(state.compareMetric)) {
    state.compareMetric = metrics[0] ?? "selection_macro_f1";
  }
  const visibleRunIds = new Set(rows.map((row) => row.run_id));
  state.compareRunIds = state.compareRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
}

export function renderComparePage(elements, rows, state, bundle) {
  renderMetricPicker(elements, state, bundle);
  renderRunPicker(elements, rows, state);
  renderSelectedRunCards(elements, rows, state);
  renderComparisonChart(elements, rows, state, bundle);
}

function renderMetricPicker(elements, state, bundle) {
  const metrics = centralEpochMetricKeys(bundle);
  elements.metricPicker.innerHTML =
    metrics.length === 0
      ? `<p class="empty">선택 가능한 step metric이 없습니다.</p>`
      : metrics
          .map(
            (metric) => `
              <label class="check-row compact">
                <input
                  type="radio"
                  name="central-compare-step-metric"
                  data-comparison-step-metric="${escapeHtml(metric)}"
                  ${metric === state.compareMetric ? "checked" : ""}
                />
                <span>${escapeHtml(metricLabel(metric))}</span>
              </label>
            `,
          )
          .join("");
}

function renderRunPicker(elements, rows, state) {
  const selectedRunIds = new Set(state.compareRunIds);
  elements.comparisonRunCheckboxes.innerHTML =
    rows.length === 0
      ? `<p class="empty">현재 필터에 해당하는 run이 없습니다.</p>`
      : rows
          .map(
            (row) => `
              <label class="run-option" title="${escapeHtml(runDetail(row))}">
                <input
                  type="checkbox"
                  data-run-id="${escapeHtml(row.run_id)}"
                  ${selectedRunIds.has(row.run_id) ? "checked" : ""}
                />
                <span>
                  <strong>${escapeHtml(algorithmName(row))}</strong>
                  <small>${escapeHtml(runDescriptor(row))}</small>
                </span>
              </label>
            `,
          )
          .join("");
}

function renderSelectedRunCards(elements, rows, state) {
  const rowsById = new Map(rows.map((row) => [row.run_id, row]));
  const selectedRows = state.compareRunIds
    .map((runId) => rowsById.get(runId))
    .filter(Boolean);
  if (selectedRows.length === 0) {
    elements.selectedRunCards.innerHTML =
      `<p class="empty">선택된 비교 run이 없습니다.</p>`;
    return;
  }
  elements.selectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const label = algorithmName(row);
      const detail = runDetail(row);
      return renderSelectedRunCard({
        id: row.run_id,
        label,
        detail,
        aliasValue: state.compareRunAliases[row.run_id],
        aliasPlaceholder: "legend alias",
        aliasDataAttribute: "comparison-alias-run-id",
        aliasAriaLabel: `${algorithmName(row)} 범례 alias`,
        removeDataAttribute: "remove-run-id",
        removeAriaLabel: `${label} 제거`,
      });
    })
    .join("");
}

function renderComparisonChart(elements, rows, state, bundle) {
  const selectedRows = rows.filter((row) => state.compareRunIds.includes(row.run_id));
  const metric = state.compareMetric;
  elements.comparisonIncludeInitial.checked = state.compareIncludeInitial;
  if (!metric) {
    elements.comparisonChart.innerHTML = `<p class="empty">비교할 step metric을 선택하세요.</p>`;
    return;
  }
  if (selectedRows.length === 0) {
    elements.comparisonChart.innerHTML = `<p class="empty">비교할 run을 선택하세요.</p>`;
    return;
  }
  if (state.compareChartType === "line") {
    elements.comparisonChart.innerHTML = drawMetricLineComparison(selectedRows, metric, state, bundle);
    return;
  }
  elements.comparisonChart.innerHTML = drawBarComparison(selectedRows, metric, state, bundle);
}

function drawMetricLineComparison(rows, metric, state, bundle) {
  const series = rows
    .map((row) => {
      const initialPoint = state.compareIncludeInitial
        ? centralInitialPoint(bundle, row, metric, state.compareEvalSet)
        : null;
      const points = [
        ...(initialPoint ? [initialPoint] : []),
        ...centralEpochPoints(bundle, row.run_id, metric),
      ];
      return {
        label: compareDisplayLabel(row, state.compareRunAliases),
        colorKey: row.run_id,
        points,
      };
    })
    .filter((item) => item.points.length > 0);
  return drawLineChart({
    series,
    metric,
    scope: "central_compare",
    xKey: "step",
    xLabel: (point) => `step ${point.step}`,
    xTickFormatter: formatStepTick,
    colorOverrides: state.compareRunColors,
    axisLabel: state.compareAxisLabel || metricLabel(metric),
    storedAxisLabel: state.compareAxisLabel,
    width: 1040,
    height: 460,
  });
}

function drawBarComparison(rows, metric, state, bundle) {
  const records = rows
    .map((row) => ({
      row,
      label: compareDisplayLabel(row, state.compareRunAliases),
      value: numberOrNull(centralLatestMetricValue(bundle, row.run_id, metric)),
    }))
    .filter((record) => record.value !== null);
  if (records.length === 0) {
    return `<p class="empty">선택한 run에 ${escapeHtml(metricLabel(metric))} step history가 없습니다.</p>`;
  }
  const max = Math.max(...records.map((record) => record.value), 0.000001);
  return records
    .map((record) => {
      const width = Math.max(2, (record.value / max) * 100);
      return `
        <div class="comparison-run">
          <div>
            <strong>${escapeHtml(record.label)}</strong>
            <span>${escapeHtml(algorithmName(record.row))}</span>
          </div>
          <div class="comparison-metrics">
            <div class="metric-bar">
              <span>${escapeHtml(metricLabel(metric))}</span>
              <div class="bar-track"><i style="width:${width}%"></i></div>
              <strong>${formatMetric(record.value)}</strong>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}
