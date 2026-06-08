import { escapeHtml } from "../../../shared/formatting/html.js";
import { metricLabel } from "../../../shared/formatting/metrics.js";
import { formatMetric } from "../../../shared/formatting/numbers.js";
import { renderCheckboxList } from "../../../ui/controls/form_controls.js";
import { emptyTableRow } from "../../../ui/tables/table.js";
import { centralOverviewMetricKeys } from "../logic/metrics.js";
import {
  algorithmName,
  overviewDisplayLabel,
  overviewRunLabel,
  overviewRunSubLabel,
  runDetail,
} from "../logic/labels.js";

export function normalizeOverviewSelection(rows, state) {
  const availableMetrics = centralOverviewMetricKeys(rows);
  state.overviewMetricIds = state.overviewMetricIds.filter((metric) =>
    availableMetrics.includes(metric),
  );
  const visibleRunIds = new Set(rows.map((row) => row.run_id));
  state.overviewRunIds = state.overviewRunIds.filter((runId) =>
    visibleRunIds.has(runId),
  );
}

export function renderOverviewPage(elements, rows, state) {
  const availableMetrics = centralOverviewMetricKeys(rows);
  renderCheckboxList(
    elements.overviewMetricPicker,
    availableMetrics,
    new Set(state.overviewMetricIds),
    "overviewMetric",
    metricLabel,
  );
  renderRunPicker(elements, rows, state);
  renderSelectedRunCards(elements, rows, state);
  renderOverviewTable(elements, rows, state);
}

function renderRunPicker(elements, rows, state) {
  const selectedRunIds = new Set(state.overviewRunIds);
  elements.overviewRunCheckboxes.innerHTML =
    rows.length === 0
      ? `<p class="empty">선택 가능한 중앙 SSL run이 없습니다.</p>`
      : rows
          .map((row) => {
            const label = overviewDisplayLabel(row, state.overviewRunAliases);
            return `
              <label class="run-option" title="${escapeHtml(runDetail(row))}">
                <input
                  type="checkbox"
                  data-overview-run-id="${escapeHtml(row.run_id)}"
                  ${selectedRunIds.has(row.run_id) ? "checked" : ""}
                />
                <span>
                  <strong>${escapeHtml(label)}</strong>
                  <small>${escapeHtml(overviewRunSubLabel(row))}</small>
                </span>
              </label>
            `;
          })
          .join("");
}

function renderSelectedRunCards(elements, rows, state) {
  const rowsById = new Map(rows.map((row) => [row.run_id, row]));
  const selectedRows = state.overviewRunIds
    .map((runId) => rowsById.get(runId))
    .filter(Boolean);
  if (selectedRows.length === 0) {
    elements.overviewSelectedRunCards.innerHTML =
      `<p class="empty">선택된 중앙 SSL run이 없습니다.</p>`;
    return;
  }
  elements.overviewSelectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const label = overviewDisplayLabel(row, state.overviewRunAliases);
      return `
        <article class="selected-run-card alias-run-card">
          <strong>${escapeHtml(label)}</strong>
          <input
            type="text"
            data-overview-alias-run-id="${escapeHtml(row.run_id)}"
            value="${escapeHtml(state.overviewRunAliases[row.run_id] ?? "")}"
            placeholder="run alias"
            aria-label="${escapeHtml(overviewRunLabel(row))} 표시명 alias"
          />
          <button
            type="button"
            data-remove-overview-run-id="${escapeHtml(row.run_id)}"
            aria-label="${escapeHtml(label)} 제거"
          >x</button>
          <span class="selected-run-detail" aria-hidden="true">${escapeHtml(runDetail(row))}</span>
        </article>
      `;
    })
    .join("");
}

function renderOverviewTable(elements, rows, state) {
  const rowsById = new Map(rows.map((row) => [row.run_id, row]));
  const selectedRows = state.overviewRunIds
    .map((runId) => rowsById.get(runId))
    .filter(Boolean);
  elements.runTableHead.innerHTML = `
    <tr>
      <th>run</th>
      <th>algorithm</th>
      ${state.overviewMetricIds.map((metric) => `<th>${escapeHtml(metricLabel(metric))}</th>`).join("")}
      <th>detail</th>
    </tr>
  `;
  if (selectedRows.length === 0) {
    elements.runTable.innerHTML = emptyTableRow(state.overviewMetricIds.length + 3, "선택된 run이 없습니다.");
    return;
  }
  elements.runTable.innerHTML = selectedRows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(overviewDisplayLabel(row, state.overviewRunAliases))}</td>
          <td>${escapeHtml(algorithmName(row))}</td>
          ${state.overviewMetricIds.map((metric) => `<td>${formatMetric(row[metric])}</td>`).join("")}
          <td>${escapeHtml(runDetail(row))}</td>
        </tr>
      `,
    )
    .join("");
}
