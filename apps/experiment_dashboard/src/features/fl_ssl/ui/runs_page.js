import { escapeHtml } from "../../../shared/formatting/html.js";
import { metricLabel } from "../../../shared/formatting/metrics.js";
import { renderCheckboxList } from "../../../ui/controls/form_controls.js";
import { emptyTableRow } from "../../../ui/tables/table.js";
import { algorithmName, compactRunSubLabel, runDescriptor, runDisplayLabel, runId } from "../logic/labels.js";
import { flRunMetricKeys, formatFlRunMetric } from "../logic/metrics.js";

export function normalizeFlRunSelection(rows, state) {
  const metrics = flRunMetricKeys(rows);
  state.runMetricIds = state.runMetricIds.filter((metric) => metrics.includes(metric));
  const visibleRunIds = new Set(rows.map(runId));
  state.runIds = state.runIds.filter((selectedRunId) =>
    visibleRunIds.has(selectedRunId),
  );
}

export function renderFlRunsPage(elements, rows, state) {
  const metrics = flRunMetricKeys(rows);
  renderCheckboxList(
    elements.flRunMetricPicker,
    metrics,
    new Set(state.runMetricIds),
    "flRunMetric",
    metricLabel,
  );
  renderRunPicker(elements, rows, state);
  renderSelectedRunCards(elements, rows, state);
  renderRunTable(elements, rows, state);
}

function renderRunPicker(elements, rows, state) {
  const selectedRunIds = new Set(state.runIds);
  elements.flRunCheckboxes.innerHTML =
    rows.length === 0
      ? `<p class="empty">선택 가능한 FL run이 없습니다.</p>`
      : rows
          .map((row) => {
            const id = runId(row);
            return `
              <label class="run-option" title="${escapeHtml(runDescriptor(row))}">
                <input
                  type="checkbox"
                  data-fl-run-id="${escapeHtml(id)}"
                  ${selectedRunIds.has(id) ? "checked" : ""}
                />
                <span>
                  <strong>${escapeHtml(runDisplayLabel(row, state.runAliases))}</strong>
                  <small>${escapeHtml(compactRunSubLabel(row))}</small>
                </span>
              </label>
            `;
          })
          .join("");
}

function renderSelectedRunCards(elements, rows, state) {
  const rowsById = new Map(rows.map((row) => [runId(row), row]));
  const selectedRows = state.runIds.map((id) => rowsById.get(id)).filter(Boolean);
  if (selectedRows.length === 0) {
    elements.flRunSelectedRunCards.innerHTML =
      `<p class="empty">선택된 FL run이 없습니다.</p>`;
    return;
  }
  elements.flRunSelectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const id = runId(row);
      const label = runDisplayLabel(row, state.runAliases);
      return `
        <article class="selected-run-card alias-run-card">
          <strong>${escapeHtml(label)}</strong>
          <input
            type="text"
            data-fl-run-alias-run-id="${escapeHtml(id)}"
            value="${escapeHtml(state.runAliases[id] ?? "")}"
            placeholder="run alias"
            aria-label="${escapeHtml(label)} 표시명 alias"
          />
          <button
            type="button"
            data-remove-fl-run-id="${escapeHtml(id)}"
            aria-label="${escapeHtml(label)} 제거"
          >x</button>
          <span class="selected-run-detail" aria-hidden="true">${escapeHtml(runDescriptor(row))}</span>
        </article>
      `;
    })
    .join("");
}

function renderRunTable(elements, rows, state) {
  const rowsById = new Map(rows.map((row) => [runId(row), row]));
  const selectedRows = state.runIds.map((id) => rowsById.get(id)).filter(Boolean);
  elements.flRunTableHead.innerHTML = `
    <tr>
      <th>run</th>
      <th>algorithm</th>
      ${state.runMetricIds.map((metric) => `<th>${escapeHtml(metricLabel(metric))}</th>`).join("")}
      <th>detail</th>
    </tr>
  `;
  if (selectedRows.length === 0) {
    elements.flRunTable.innerHTML = emptyTableRow(state.runMetricIds.length + 3, "선택된 FL run이 없습니다.");
    return;
  }
  elements.flRunTable.innerHTML = selectedRows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(runDisplayLabel(row, state.runAliases))}</td>
          <td>${escapeHtml(algorithmName(row))}</td>
          ${state.runMetricIds.map((metric) => `<td>${formatFlRunMetric(row, metric)}</td>`).join("")}
          <td>${escapeHtml(runDescriptor(row))}</td>
        </tr>
      `,
    )
    .join("");
}
