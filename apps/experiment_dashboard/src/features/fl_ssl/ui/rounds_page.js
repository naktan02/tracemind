import { escapeHtml } from "../../../shared/formatting/html.js";
import { metricLabel } from "../../../shared/formatting/metrics.js";
import { formatBytes, formatMetric, formatSeconds, numberOrNull } from "../../../shared/formatting/numbers.js";
import { drawLineChart } from "../../../ui/charts/line_chart.js";
import { emptyTableRow } from "../../../ui/tables/table.js";
import { FL_ROUND_METRICS } from "../logic/constants.js";
import { algorithmName, roundLegendLabel, runDescriptor, runId } from "../logic/labels.js";
import { roundPointValue } from "../logic/metrics.js";
import { compareFlRoundRows, flRoundRows } from "../logic/selectors.js";

export function normalizeRoundSelection(rows, state) {
  if (!FL_ROUND_METRICS.includes(state.roundMetric)) {
    state.roundMetric = "macro_f1";
  }
  const visibleRunIds = new Set(rows.map(runId));
  state.roundRunIds = state.roundRunIds.filter((id) => visibleRunIds.has(id));
}

export function renderRoundsPage(elements, rows, state, bundle) {
  renderRunPicker(elements, rows, state);
  renderSelectedRunCards(elements, rows, state);
  renderMetricTabs(elements, state);
  const selectedRows = flRoundRows(bundle)
    .filter((row) => state.roundRunIds.includes(row.run_id))
    .filter((row) => state.roundIncludeInitial || numberOrNull(row.round_index) !== 0)
    .sort(compareFlRoundRows);
  renderFlatNote(elements, selectedRows, state);
  renderRoundChart(elements, selectedRows, rows, state);
  renderRoundTable(elements, selectedRows, state);
}

function renderRunPicker(elements, rows, state) {
  const selectedRunIds = new Set(state.roundRunIds);
  elements.flRoundRunCheckboxes.innerHTML =
    rows.length === 0
      ? `<p class="empty">선택 가능한 run이 없습니다.</p>`
      : rows
          .map((row) => {
            const id = runId(row);
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
  elements.flRoundSelectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const id = runId(row);
      const label = roundLegendLabel(row, state.roundRunAliases);
      return `
        <article class="selected-run-card alias-run-card">
          <strong>${escapeHtml(label)}</strong>
          <input
            type="text"
            data-fl-round-alias-run-id="${escapeHtml(id)}"
            value="${escapeHtml(state.roundRunAliases[id] ?? "")}"
            placeholder="legend alias"
            aria-label="${escapeHtml(algorithmName(row))} 범례 alias"
          />
          <button
            type="button"
            data-remove-fl-round-run-id="${escapeHtml(id)}"
            aria-label="${escapeHtml(label)} 제거"
          >x</button>
          <span class="selected-run-detail" aria-hidden="true">${escapeHtml(runDescriptor(row))}</span>
        </article>
      `;
    })
    .join("");
}

function renderMetricTabs(elements, state) {
  elements.flRoundMetricPicker.innerHTML = FL_ROUND_METRICS.map(
    (metric) => `
      <button
        type="button"
        data-fl-round-metric="${escapeHtml(metric)}"
        class="${metric === state.roundMetric ? "active" : ""}"
      >${escapeHtml(metricLabel(metric))}</button>
    `,
  ).join("");
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

function renderRoundTable(elements, rows, state) {
  if (rows.length === 0) {
    elements.flRoundTable.innerHTML = emptyTableRow(13, "선택한 run의 round metric이 없습니다.");
    return;
  }
  elements.flRoundTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(roundTableLabel(row, state))}</td>
          <td>${formatMetric(row.macro_f1)}</td>
          <td>${formatMetric(row.accuracy_top_1)}</td>
          <td>${formatMetric(row.loss)}</td>
          <td>${formatMetric(row.expected_calibration_error)}</td>
          <td>${formatMetric(row.accepted_ratio)}</td>
          <td>${formatMetric(row.update_count)}</td>
          <td>${formatBytes(row.total_payload_bytes)}</td>
          <td>${formatSeconds(row.round_time_seconds)}</td>
          <td>${formatMetric(row.gpu_memory_peak_mb)}</td>
          <td>${formatMetric(row.round_update_delta_l2_mean)}</td>
          <td>${formatMetric(row.round_update_delta_l2_max)}</td>
          <td>${formatMetric(row.round_update_cosine_to_mean_mean)}</td>
        </tr>
      `,
    )
    .join("");
}

function roundTableLabel(row, state) {
  const base = row.round_id ?? roundLabel(row.round_index);
  return state.roundRunIds.length <= 1 ? base : `${row.run_id} · ${base}`;
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
