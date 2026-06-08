import { escapeHtml } from "../../../shared/formatting/html.js";
import { fillSelect } from "../../../ui/controls/form_controls.js";
import {
  emptyTableRow,
  moveTableColumn,
  renderSortableTableHeader,
  resolveTableColumns,
} from "../../../ui/tables/table.js";
import { runDetailLabel, runId } from "../logic/labels.js";
import { flRunsWithRows, flSplitRows } from "../logic/selectors.js";

const SPLIT_COLUMNS = [
  { id: "client", label: "client", render: (row) => escapeHtml(row.client_id) },
  { id: "labeled", label: "labeled", render: (row) => escapeHtml(row.labeled_count ?? "-") },
  { id: "unlabeled", label: "unlabeled", render: (row) => escapeHtml(row.unlabeled_count ?? "-") },
  {
    id: "labeled_dist",
    label: "labeled dist",
    render: (row) => escapeHtml(formatDistribution(row.labeled_label_distribution)),
  },
  {
    id: "unlabeled_dist",
    label: "unlabeled dist",
    render: (row) => escapeHtml(formatDistribution(row.unlabeled_label_distribution)),
  },
  {
    id: "combined_dist",
    label: "combined dist",
    render: (row) => escapeHtml(formatDistribution(row.combined_label_distribution)),
  },
];

export function normalizeSplitSelection(rows, state, bundle) {
  const candidateRuns = flRunsWithRows(rows, flSplitRows(bundle));
  const ids = new Set(candidateRuns.map(runId));
  if (!state.splitRunId || !ids.has(state.splitRunId)) {
    state.splitRunId = candidateRuns.length > 0 ? runId(candidateRuns[0]) : null;
  }
}

export function renderSplitsPage(elements, rows, state, bundle, rerender = () => {}) {
  const candidateRuns = flRunsWithRows(rows, flSplitRows(bundle));
  fillSelect(elements.flSplitRunFilter, candidateRuns.map(runId), state.splitRunId, "run 없음");
  Array.from(elements.flSplitRunFilter.options).forEach((option) => {
    const row = candidateRuns.find((candidate) => runId(candidate) === option.value);
    if (row) option.textContent = runDetailLabel(row);
  });
  const tableRows = flSplitRows(bundle)
    .filter((row) => row.run_id === state.splitRunId)
    .sort((left, right) => String(left.client_id ?? "").localeCompare(String(right.client_id ?? "")));

  const { visibleColumns } = resolveTableColumns(
    state.splitTableColumns,
    SPLIT_COLUMNS,
    SPLIT_COLUMNS.map((column) => column.id),
  );
  renderSortableTableHeader(
    elements.flSplitTableHead,
    visibleColumns,
    (sourceColumnId, targetColumnId) => {
      if (moveTableColumn(state.splitTableColumns, sourceColumnId, targetColumnId)) {
        rerender();
      }
    },
  );
  elements.flSplitTable.innerHTML =
    tableRows.length === 0
      ? emptyTableRow(visibleColumns.length || 1, "split 정보가 없습니다.")
      : tableRows
          .map(
            (row) => `
              <tr>
                ${visibleColumns
                  .map((column) => `<td>${column.render(row)}</td>`)
                  .join("")}
              </tr>
            `,
          )
          .join("");
}

function formatDistribution(value) {
  if (!value) return "-";
  if (typeof value === "string") return value;
  return Object.entries(value)
    .map(([label, count]) => `${label}:${count}`)
    .join(", ");
}
