import { escapeHtml } from "../../../shared/formatting/html.js";
import { metricLabel } from "../../../shared/formatting/metrics.js";
import { formatMetric } from "../../../shared/formatting/numbers.js";
import {
  emptyTableRow,
  moveTableColumn,
  renderColumnCheckboxes,
  renderSortableTableHeader,
  resolveTableColumns,
} from "../../../ui/tables/table.js";
import { renderSelectedRunCard } from "../../../ui/controls/selected_run_card.js";
import { renderRunOptionDetail } from "../../../ui/controls/run_option_detail.js";
import { CENTRAL_FILTER_AXES } from "../logic/filters.js";
import { centralOverviewMetricKeys } from "../logic/metrics.js";
import {
  algorithmName,
  overviewRunLabel,
  overviewRunSubLabel,
  runDetail,
} from "../logic/labels.js";

const DEFAULT_VISIBLE_COLUMNS = [
  "axis:algorithm",
];

const OVERVIEW_AXIS_COLUMNS = [
  {
    id: "axis:algorithm",
    label: "algorithm",
    group: "axis",
    render: (row, state) =>
      escapeHtml(state.overviewRunAliases[row.run_id] || algorithmName(row)),
  },
  {
    id: "axis:run",
    label: "run",
    group: "axis",
    render: (row) => escapeHtml(overviewRunLabel(row)),
  },
  {
    id: "axis:pc",
    label: "pc",
    group: "axis",
    render: (row) => escapeHtml(overviewRunSubLabel(row)),
  },
  {
    id: "axis:seed",
    label: "seed",
    group: "axis",
    render: (row) => escapeHtml(row.seed ?? "-"),
  },
  {
    id: "axis:run_id",
    label: "run id",
    group: "axis",
    render: (row) => escapeHtml(row.run_id),
  },
  {
    id: "axis:detail",
    label: "detail",
    group: "axis",
    render: (row) => escapeHtml(runDetail(row)),
  },
];

export function normalizeOverviewSelection(rows, state) {
  const availableMetrics = centralOverviewMetricKeys(rows);
  const availableColumnIds = buildOverviewColumns(rows).map((column) => column.id);
  state.overviewMetricIds = state.overviewMetricIds.filter((metric) =>
    availableMetrics.includes(metric),
  );
  normalizeOverviewColumns(state, availableColumnIds);
}

export function renderOverviewPage(elements, rows, state, _bundle, rerender = () => {}, selectionRows = rows) {
  const columns = buildOverviewColumns(rows);
  const { visibleColumns, allColumns, state: columnState } = resolveTableColumns(
    state.overviewTableColumns,
    columns,
    DEFAULT_VISIBLE_COLUMNS,
  );

  const axisColumns = allColumns.filter((column) => column.group === "axis");
  const metricColumns = allColumns.filter((column) => column.group === "metric");
  const visibleIds = new Set(columnState.visible);

  renderColumnCheckboxes(
    elements.overviewMetricPicker,
    metricColumns,
    visibleIds,
    "overviewTableColumn",
  );
  renderColumnCheckboxes(
    elements.overviewAxisPicker,
    axisColumns,
    visibleIds,
    "overviewTableColumn",
  );
  renderRunPicker(elements, rows, state);
  renderSelectedRunCards(elements, selectionRows, state);
  renderOverviewTable(elements, visibleColumns, selectionRows, state, rerender);
}

function renderRunPicker(elements, rows, state) {
  const selectedRunIds = new Set(state.overviewRunIds);
  const peerDetails = rows.map(runDetail);
  elements.overviewRunCheckboxes.innerHTML =
    rows.length === 0
      ? `<p class="empty">선택 가능한 run이 없습니다.</p>`
      : rows
          .map((row) => {
            const label = overviewRunLabel(row);
            const detail = runDetail(row);
            return `
              <label class="run-option">
                <input
                  type="checkbox"
                  data-overview-run-id="${escapeHtml(row.run_id)}"
                  ${selectedRunIds.has(row.run_id) ? "checked" : ""}
                />
                <span>
                  <strong>${escapeHtml(label)}</strong>
                  <small>${escapeHtml(overviewRunSubLabel(row))}</small>
                </span>
                <span class="run-option-detail" aria-hidden="true">${renderRunOptionDetail(detail, peerDetails)}</span>
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
      `<p class="empty">선택된 run이 없습니다.</p>`;
    return;
  }
  const peerDetails = selectedRows.map(runDetail);
  elements.overviewSelectedRunCards.innerHTML = selectedRows
    .map((row) => {
      const label = overviewRunLabel(row);
      const detail = runDetail(row);
      return renderSelectedRunCard({
        id: row.run_id,
        label,
        detail,
        peerDetails,
        aliasValue: state.overviewRunAliases[row.run_id],
        aliasPlaceholder: "run alias",
        aliasDataAttribute: "overview-alias-run-id",
        aliasAriaLabel: `${overviewRunLabel(row)} 표시명 alias`,
        removeDataAttribute: "remove-overview-run-id",
        removeAriaLabel: `${label} 제거`,
      });
    })
    .join("");
}

function renderOverviewTable(elements, columns, rows, state, rerender) {
  const rowsById = new Map(rows.map((row) => [row.run_id, row]));
  const selectedRows = state.overviewRunIds
    .map((runId) => rowsById.get(runId))
    .filter(Boolean);
  renderSortableTableHeader(elements.runTableHead, columns, (sourceColumnId, targetColumnId) => {
    if (moveTableColumn(state.overviewTableColumns, sourceColumnId, targetColumnId)) {
      rerender();
    }
  });

  if (selectedRows.length === 0) {
    elements.runTable.innerHTML = emptyTableRow(columns.length || 1, "선택된 run이 없습니다.");
    return;
  }
  elements.runTable.innerHTML = selectedRows
    .map(
      (row) => `
        <tr>
          ${columns.map((column) => `<td>${column.render(row, state)}</td>`).join("")}
        </tr>
      `,
    )
    .join("");
}

function buildOverviewColumns(rows) {
  const metricColumns = centralOverviewMetricKeys(rows).map((metric) => ({
    id: `metric:${metric}`,
    group: "metric",
    label: metricLabel(metric),
    render: (row) => formatMetric(row[metric]),
  }));
  return [...OVERVIEW_AXIS_COLUMNS, ...buildOverviewAxisColumns(), ...metricColumns];
}

function buildOverviewAxisColumns() {
  const existingIds = new Set(OVERVIEW_AXIS_COLUMNS.map((column) => column.id));
  const dynamicAxisColumns = CENTRAL_FILTER_AXES
    .filter((axis) => !existingIds.has(`axis:${axis.id}`))
    .map((axis) => ({
      id: `axis:${axis.id}`,
      group: "axis",
      label: axis.label,
      render: (row) =>
        escapeHtml(axis.labelForValue ? axis.labelForValue(row) : axis.value(row)),
    }));
  return dynamicAxisColumns;
}

function normalizeOverviewColumns(state, availableColumnIds) {
  const availableSet = new Set(availableColumnIds);
  const tableState = state.overviewTableColumns;
  const visible = (tableState.visible ?? []).filter((id) => availableSet.has(id));
  const visibleSet = new Set(visible);
  const ordered = (tableState.order ?? []).filter(
    (id) => availableSet.has(id) && visibleSet.has(id),
  );
  for (const id of visible) {
    if (!ordered.includes(id)) {
      ordered.push(id);
    }
  }
  tableState.visible = visible;
  tableState.order = ordered;
  state.overviewMetricIds = visible
    .filter((id) => id.startsWith("metric:"))
    .map((id) => id.replace(/^metric:/, ""));
}
