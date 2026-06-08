import { escapeHtml } from "../../../shared/formatting/html.js";
import { metricLabel } from "../../../shared/formatting/metrics.js";
import {
  emptyTableRow,
  moveTableColumn,
  renderColumnCheckboxes,
  renderSortableTableHeader,
  resolveTableColumns,
  setTableColumnVisibility,
} from "../../../ui/tables/table.js";
import {
  algorithmName,
  compactRunSubLabel,
  labelBudgetLabel,
  runDescriptor,
  runDisplayLabel,
  runId,
} from "../logic/labels.js";
import { flRunMetricKeys, formatFlRunMetric } from "../logic/metrics.js";

const DEFAULT_VISIBLE_COLUMNS = ["axis:algorithm"];

const FL_RUN_AXIS_COLUMNS = [
  {
    id: "axis:algorithm",
    group: "axis",
    label: "algorithm",
    render: (row) => escapeHtml(algorithmName(row)),
  },
  {
    id: "axis:run",
    group: "axis",
    label: "run",
    render: (row, state) => escapeHtml(runDisplayLabel(row, state.runAliases)),
  },
  {
    id: "axis:pc",
    group: "axis",
    label: "pc",
    render: (row) => escapeHtml(labelBudgetLabel(row)),
  },
  {
    id: "axis:clients",
    group: "axis",
    label: "clients",
    render: (row) => escapeHtml(row.client_count ?? "-"),
  },
  {
    id: "axis:rank",
    group: "axis",
    label: "rank",
    render: (row) => escapeHtml(row.peft_adapter_rank ?? "-"),
  },
  {
    id: "axis:seed",
    group: "axis",
    label: "seed",
    render: (row) => escapeHtml(row.seed ?? "-"),
  },
  {
    id: "axis:detail",
    group: "axis",
    label: "detail",
    render: (row) => escapeHtml(runDescriptor(row)),
  },
];

export function normalizeFlRunSelection(rows, state) {
  const metrics = flRunMetricKeys(rows);
  state.runMetricIds = state.runMetricIds.filter((metric) => metrics.includes(metric));
  const visibleRunIds = new Set(rows.map(runId));
  state.runIds = state.runIds.filter((selectedRunId) =>
    visibleRunIds.has(selectedRunId),
  );
  normalizeRunColumns(state, rows);
}

export function renderFlRunsPage(elements, rows, state, rerender = () => {}) {
  const columns = buildRunColumns(rows);
  const { visibleColumns, allColumns, state: columnState } = resolveTableColumns(
    state.runTableColumns,
    columns,
    DEFAULT_VISIBLE_COLUMNS,
  );

  const axisColumns = allColumns.filter((column) => column.group === "axis");
  const metricColumns = allColumns.filter((column) => column.group === "metric");
  const visibleIds = new Set(columnState.visible);

  renderColumnCheckboxes(elements.flRunMetricPicker, metricColumns, visibleIds, "flRunTableColumn");
  renderColumnCheckboxes(elements.flRunAxisPicker, axisColumns, visibleIds, "flRunTableColumn");
  renderRunPicker(elements, rows, state);
  renderSelectedRunCards(elements, rows, state);
  renderRunTable(elements, visibleColumns, rows, state, rerender);
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

function renderRunTable(elements, columns, rows, state, rerender) {
  const rowsById = new Map(rows.map((row) => [runId(row), row]));
  const selectedRows = state.runIds.map((id) => rowsById.get(id)).filter(Boolean);
  renderSortableTableHeader(elements.flRunTableHead, columns, (sourceColumnId, targetColumnId) => {
    if (moveTableColumn(state.runTableColumns, sourceColumnId, targetColumnId)) {
      rerender();
    }
  });
  if (selectedRows.length === 0) {
    elements.flRunTable.innerHTML = emptyTableRow(columns.length || 1, "선택된 FL run이 없습니다.");
    return;
  }
  elements.flRunTable.innerHTML = selectedRows
    .map(
      (row) => `
        <tr>
          ${columns.map((column) => `<td>${column.render(row, state)}</td>`).join("")}
        </tr>
      `,
    )
    .join("");
}

function buildRunColumns(rows) {
  const metricColumns = flRunMetricKeys(rows).map((metric) => ({
    id: `metric:${metric}`,
    group: "metric",
    label: metricLabel(metric),
    render: (row) => formatFlRunMetric(row, metric),
  }));
  return [...FL_RUN_AXIS_COLUMNS, ...metricColumns];
}

function normalizeRunColumns(state, rows) {
  const columns = buildRunColumns(rows);
  const visibleCandidates = (state.runMetricIds ?? [])
    .map((metric) => `metric:${metric}`)
    .filter((id) => columns.some((column) => column.id === id));
  const fallback = visibleCandidates.length > 0 ? visibleCandidates : DEFAULT_VISIBLE_COLUMNS;
  setTableColumnVisibility(state.runTableColumns, columns, fallback, DEFAULT_VISIBLE_COLUMNS);
}
