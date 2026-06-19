import { escapeHtml } from "../../../shared/formatting/html.js";
import { metricLabel } from "../../../shared/formatting/metrics.js";
import { formatBytes, formatMetric, formatSeconds, numberOrNull } from "../../../shared/formatting/numbers.js";
import { drawLineChart } from "../../../ui/charts/line_chart.js";
import {
  emptyTableRow,
  moveTableColumn,
  renderSortableTableHeader,
  resolveTableColumns,
  setTableColumnVisibility,
} from "../../../ui/tables/table.js";
import { renderSelectedRunCard } from "../../../ui/controls/selected_run_card.js";
import { renderRunOptionDetail } from "../../../ui/controls/run_option_detail.js";
import { FL_ROUND_METRICS } from "../logic/constants.js";
import {
  algorithmName,
  roundLegendLabel,
  runDescriptor,
  runHoverDetail,
  runId,
} from "../logic/labels.js";
import { roundPointValue } from "../logic/metrics.js";
import { compareFlRoundRows, flRoundRows } from "../logic/selectors.js";

const ROUND_TABLE_COLUMNS = [
  {
    id: "axis:round",
    group: "axis",
    label: "round",
    render: (row) => escapeHtml(roundLabel(row.round_index)),
  },
  {
    id: "axis:run",
    group: "axis",
    label: "run",
    render: (row) => escapeHtml(row.run_id),
  },
];
const ROUND_BASE_VISIBLE_COLUMNS = ["axis:round", "axis:run"];
const ROUND_DEFAULT_METRIC = "macro_f1";

export function normalizeRoundSelection(rows, state) {
  const metricColumns = FL_ROUND_METRICS.map((metric) => `metric:${metric}`);
  const requestedMetric = `metric:${FL_ROUND_METRICS.includes(state.roundMetric) ? state.roundMetric : ROUND_DEFAULT_METRIC}`;
  const fallbackMetric = metricColumns.includes(`metric:${ROUND_DEFAULT_METRIC}`)
    ? `metric:${ROUND_DEFAULT_METRIC}`
    : metricColumns[0];
  const selectedMetricId = metricColumns.includes(requestedMetric) ? requestedMetric : fallbackMetric;
  const fallbackVisible = [selectedMetricId].filter(Boolean);
  const requestedVisibleColumns = [...ROUND_BASE_VISIBLE_COLUMNS, ...fallbackVisible];
  setTableColumnVisibility(state.roundTableColumns, buildRoundColumns(), requestedVisibleColumns, ROUND_BASE_VISIBLE_COLUMNS);
  state.roundMetric = state.roundTableColumns.visible
    .filter((id) => id.startsWith("metric:"))
    .map((id) => id.replace(/^metric:/, ""))[0] || ROUND_DEFAULT_METRIC;
  state.roundMetricIds = [state.roundMetric];
}

export function renderRoundsPage(elements, rows, state, bundle, rerender = () => {}, selectionRows = rows) {
  const columns = buildRoundColumns();
  const { visibleColumns, allColumns } = resolveTableColumns(
    state.roundTableColumns,
    columns,
    ROUND_BASE_VISIBLE_COLUMNS,
  );
  const metricColumns = allColumns.filter((column) => column.group === "metric");
  renderRoundMetricPicker(elements.flRoundTableMetricPicker, metricColumns, state.roundMetric);
  renderRunPicker(elements, rows, state);
  renderSelectedRunCards(elements, selectionRows, state);
  const selectedRows = flRoundRows(bundle)
    .filter((row) => state.roundRunIds.includes(row.run_id))
    .filter((row) => state.roundIncludeInitial || numberOrNull(row.round_index) !== 0)
    .sort(compareFlRoundRows);
  renderFlatNote(elements, selectedRows, state);
  renderRoundChart(elements, selectedRows, selectionRows, state);
  renderRoundTable(elements, columns, selectedRows, visibleColumns, state, rerender);
}

function renderRunPicker(elements, rows, state) {
  const selectedRunIds = new Set(state.roundRunIds);
  const peerDetails = rows.map(runHoverDetail);
  elements.flRoundRunCheckboxes.innerHTML =
    rows.length === 0
      ? `<p class="empty">선택 가능한 run이 없습니다.</p>`
      : rows
          .map((row) => {
            const id = runId(row);
            const detail = runHoverDetail(row);
            return `
              <label class="run-option">
                <input
                  type="checkbox"
                  data-fl-round-run-id="${escapeHtml(id)}"
                  ${selectedRunIds.has(id) ? "checked" : ""}
                />
                <span>
                  <strong>${escapeHtml(algorithmName(row))}</strong>
                  <small>${escapeHtml(runDescriptor(row))}</small>
                </span>
                <span class="run-option-detail" aria-hidden="true">${renderRunOptionDetail(detail, peerDetails)}</span>
              </label>
            `;
          })
          .join("");
}

function renderSelectedRunCards(elements, rows, state) {
  const rowsById = new Map(rows.map((row) => [runId(row), row]));
  const selectedRows = state.roundRunIds.map((id) => rowsById.get(id)).filter(Boolean);
  if (selectedRows.length === 0) {
    elements.flRoundSelectedRunCards.innerHTML =
      `<p class="empty">선택된 FL round run이 없습니다.</p>`;
    return;
  }
  const peerDetails = selectedRows.map(runHoverDetail);
  elements.flRoundSelectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const id = runId(row);
      const label = roundLegendLabel(row, state.roundRunAliases);
      const detail = runHoverDetail(row);
      return renderSelectedRunCard({
        id,
        label,
        detail,
        peerDetails,
        aliasValue: state.roundRunAliases[id],
        aliasPlaceholder: "legend alias",
        aliasDataAttribute: "fl-round-alias-run-id",
        aliasAriaLabel: `${algorithmName(row)} 범례 alias`,
        removeDataAttribute: "remove-fl-round-run-id",
        removeAriaLabel: `${label} 제거`,
      });
    })
    .join("");
}

function renderFlatNote(elements, rows, state) {
  const values = Array.from(
    new Set(
      rows
        .map((row) => numberOrNull(row[state.roundMetric]))
        .filter((value) => value !== null),
    ),
  );
  elements.flRoundFlatNote.hidden = rows.length === 0 || values.length !== 1;
  elements.flRoundFlatNote.textContent = elements.flRoundFlatNote.hidden
    ? ""
    : `선택한 run에서 ${metricLabel(state.roundMetric)} 값이 전 라운드 동일합니다.`;
}

function renderRoundMetricPicker(container, metricColumns, selectedMetric) {
  const selectedMetricId = `metric:${selectedMetric}`;
  container.innerHTML =
    metricColumns.length === 0
      ? `<p class="empty">선택 가능한 metric이 없습니다.</p>`
      : metricColumns
          .map(
            (column) => `
              <label class="check-row compact">
                <input
                  type="radio"
                  name="fl-round-metric"
                  data-fl-round-table-metric="${escapeHtml(column.id)}"
                  ${column.id === selectedMetricId ? "checked" : ""}
                />
                <span>${escapeHtml(column.label)}</span>
              </label>
            `,
          )
          .join("");
}

function renderRoundChart(elements, roundRows, runRows, state) {
  const runById = new Map(runRows.map((row) => [runId(row), row]));
  const grouped = new Map();
  for (const row of roundRows) {
    if (!grouped.has(row.run_id)) grouped.set(row.run_id, []);
    grouped.get(row.run_id).push(row);
  }
  const series = state.roundRunIds
    .filter((id) => grouped.has(id))
    .map((id) => {
      const run = runById.get(id);
      return {
        label: run ? roundLegendLabel(run, state.roundRunAliases) : id,
        colorKey: id,
        points: grouped
          .get(id)
          .slice()
          .sort((left, right) => numberOrNull(left.round_index) - numberOrNull(right.round_index))
          .map((row) => ({
            roundIndex: numberOrNull(row.round_index),
            value: roundPointValue(row, state.roundMetric),
          }))
          .filter((point) => point.roundIndex !== null && point.value !== null),
      };
    })
    .filter((item) => item.points.length > 0);
  elements.flRoundChart.innerHTML = drawLineChart({
    series,
    metric: state.roundMetric,
    scope: "fl_round",
    xKey: "roundIndex",
    xLabel: (point) => roundLabel(point.roundIndex),
    xTickFormatter: compactRoundLabel,
    colorOverrides: state.roundRunColors,
    axisLabel: state.roundAxisLabel || metricLabel(state.roundMetric),
    storedAxisLabel: state.roundAxisLabel,
    width: 1160,
    height: 520,
  });
}

function buildRoundColumns() {
  const metricColumns = FL_ROUND_METRICS.map((metric) => ({
    id: `metric:${metric}`,
    group: "metric",
    label: metricLabel(metric),
    render: (row) => formatRoundMetric(row, metric),
  }));
  return [
    ...ROUND_TABLE_COLUMNS,
    ...metricColumns,
  ];
}

function renderRoundTable(elements, columns, rows, visibleColumns, state, rerender) {
  renderSortableTableHeader(elements.flRoundTableHead, visibleColumns, (sourceColumnId, targetColumnId) => {
    if (moveTableColumn(state.roundTableColumns, sourceColumnId, targetColumnId)) {
      rerender();
    }
  });
  if (rows.length === 0) {
    elements.flRoundTable.innerHTML = emptyTableRow(
      visibleColumns.length || 1,
      "선택한 run의 round metric이 없습니다.",
    );
    return;
  }
  const rowById = new Map(columns.map((column) => [column.id, column]));
  elements.flRoundTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          ${visibleColumns
            .map((column) => `<td>${rowById.get(column.id)?.render(row)}</td>`)
            .join("")}
        </tr>
      `,
    )
    .join("");
}

function formatRoundMetric(row, metric) {
  if (metric === "macro_f1") return formatMetric(row.macro_f1);
  if (metric === "accuracy_top_1") return formatMetric(row.accuracy_top_1);
  if (metric === "loss") return formatMetric(row.loss);
  if (metric === "expected_calibration_error") return formatMetric(row.expected_calibration_error);
  if (metric === "accepted_ratio") return formatMetric(row.accepted_ratio);
  if (metric === "update_count") return formatMetric(row.update_count);
  if (metric === "total_payload_bytes") return formatBytes(row.total_payload_bytes);
  if (metric === "round_time_seconds") return formatSeconds(row.round_time_seconds);
  if (metric === "gpu_memory_peak_mb") return formatMetric(row.gpu_memory_peak_mb);
  if (metric === "macro_f1_delta_from_initial") return formatMetric(row.macro_f1_delta_from_initial);
  if (metric === "macro_f1_delta_from_previous") return formatMetric(row.macro_f1_delta_from_previous);
  if (metric === "loss_delta_from_initial") return formatMetric(row.loss_delta_from_initial);
  if (metric === "loss_delta_from_previous") return formatMetric(row.loss_delta_from_previous);
  if (metric === "ece_delta_from_initial") return formatMetric(row.ece_delta_from_initial);
  if (metric === "accepted_ratio_delta_from_initial") return formatMetric(row.accepted_ratio_delta_from_initial);
  if (metric === "round_update_delta_l2_mean") return formatMetric(row.round_update_delta_l2_mean);
  if (metric === "round_update_delta_l2_max") return formatMetric(row.round_update_delta_l2_max);
  if (metric === "round_update_delta_to_mean_l2_mean") return formatMetric(row.round_update_delta_to_mean_l2_mean);
  if (metric === "round_update_delta_to_mean_l2_max") return formatMetric(row.round_update_delta_to_mean_l2_max);
  if (metric === "round_update_cosine_to_mean_mean") return formatMetric(row.round_update_cosine_to_mean_mean);
  if (metric === "round_update_cosine_to_mean_min") return formatMetric(row.round_update_cosine_to_mean_min);
  return "-";
}

function roundLabel(roundIndex) {
  const number = numberOrNull(roundIndex);
  if (number === null) return "round ?";
  return number === 0 ? "initial" : `round ${number}`;
}

function compactRoundLabel(roundIndex) {
  const number = numberOrNull(roundIndex);
  if (number === null) return "?";
  return number === 0 ? "init" : `r${number}`;
}
