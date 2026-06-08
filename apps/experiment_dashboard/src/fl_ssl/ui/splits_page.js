import { escapeHtml } from "../../shared/formatting/html.js";
import { fillSelect } from "../../ui/controls/form_controls.js";
import { emptyTableRow } from "../../ui/tables/table.js";
import { runDetailLabel, runId } from "../logic/labels.js";
import { flRunsWithRows, flSplitRows } from "../logic/selectors.js";

export function normalizeSplitSelection(rows, state, bundle) {
  const candidateRuns = flRunsWithRows(rows, flSplitRows(bundle));
  const ids = new Set(candidateRuns.map(runId));
  if (!state.splitRunId || !ids.has(state.splitRunId)) {
    state.splitRunId = candidateRuns.length > 0 ? runId(candidateRuns[0]) : null;
  }
}

export function renderSplitsPage(elements, rows, state, bundle) {
  const candidateRuns = flRunsWithRows(rows, flSplitRows(bundle));
  fillSelect(elements.flSplitRunFilter, candidateRuns.map(runId), state.splitRunId, "run 없음");
  Array.from(elements.flSplitRunFilter.options).forEach((option) => {
    const row = candidateRuns.find((candidate) => runId(candidate) === option.value);
    if (row) option.textContent = runDetailLabel(row);
  });
  const tableRows = flSplitRows(bundle)
    .filter((row) => row.run_id === state.splitRunId)
    .sort((left, right) => String(left.client_id ?? "").localeCompare(String(right.client_id ?? "")));
  elements.flSplitTable.innerHTML =
    tableRows.length === 0
      ? emptyTableRow(6, "split 정보가 없습니다.")
      : tableRows
          .map(
            (row) => `
              <tr>
                <td>${escapeHtml(row.client_id)}</td>
                <td>${escapeHtml(row.labeled_count ?? "-")}</td>
                <td>${escapeHtml(row.unlabeled_count ?? "-")}</td>
                <td>${escapeHtml(formatDistribution(row.labeled_label_distribution))}</td>
                <td>${escapeHtml(formatDistribution(row.unlabeled_label_distribution))}</td>
                <td>${escapeHtml(formatDistribution(row.combined_label_distribution))}</td>
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
