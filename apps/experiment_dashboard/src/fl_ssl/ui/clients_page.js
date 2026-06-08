import { escapeHtml } from "../../shared/formatting/html.js";
import { formatBytes, formatMetric, formatSeconds, numberOrNull } from "../../shared/formatting/numbers.js";
import { fillSelect } from "../../ui/controls/form_controls.js";
import { emptyTableRow } from "../../ui/tables/table.js";
import { runDetailLabel, runId } from "../logic/labels.js";
import {
  compareClientRoundRows,
  flClientRoundRows,
  flClientValidationRows,
  flRunsWithRows,
} from "../logic/selectors.js";

export function normalizeClientSelections(rows, state, bundle) {
  state.clientValidationRunId = normalizeRunId(
    state.clientValidationRunId,
    flRunsWithRows(rows, flClientValidationRows(bundle)),
  );
  state.clientRoundRunId = normalizeRunId(
    state.clientRoundRunId,
    flRunsWithRows(rows, flClientRoundRows(bundle)),
  );
}

export function renderClientsPage(elements, rows, state, bundle) {
  renderClientValidation(elements, rows, state, bundle);
  renderClientRounds(elements, rows, state, bundle);
}

function renderClientValidation(elements, rows, state, bundle) {
  const candidateRuns = flRunsWithRows(rows, flClientValidationRows(bundle));
  fillRunSelect(elements.flClientValidationRunFilter, candidateRuns, state.clientValidationRunId);
  const tableRows = flClientValidationRows(bundle)
    .filter((row) => row.run_id === state.clientValidationRunId)
    .sort((left, right) => String(left.client_id ?? "").localeCompare(String(right.client_id ?? "")));
  elements.flClientValidationTable.innerHTML =
    tableRows.length === 0
      ? emptyTableRow(10, "client validation 결과가 없습니다.")
      : tableRows
          .map(
            (row) => `
              <tr>
                <td>${escapeHtml(row.client_id)}</td>
                <td>${formatMetric(row.labeled_count)}</td>
                <td>${formatMetric(row.unlabeled_count)}</td>
                <td>${formatMetric(row.macro_f1)}</td>
                <td>${formatMetric(row.accuracy_top_1)}</td>
                <td>${formatMetric(row.loss)}</td>
                <td>${formatMetric(row.expected_calibration_error)}</td>
                <td>${formatMetric(row.total_accepted_pseudo_labels)}</td>
                <td>${formatMetric(row.update_round_count)}</td>
                <td>${formatMetric(row.mean_update_delta_l2)}</td>
              </tr>
            `,
          )
          .join("");
}

function renderClientRounds(elements, rows, state, bundle) {
  const candidateRuns = flRunsWithRows(rows, flClientRoundRows(bundle));
  fillRunSelect(elements.flClientRoundRunFilter, candidateRuns, state.clientRoundRunId);
  const runRows = flClientRoundRows(bundle).filter(
    (row) => row.run_id === state.clientRoundRunId,
  );
  fillRoundSelect(elements.flClientRoundFilter, runRows, state.clientRoundIndex);
  const selectedRound = selectedClientRoundIndex(runRows, state);
  const tableRows = runRows
    .filter((row) => numberOrNull(row.round_index) === selectedRound)
    .sort(compareClientRoundRows);
  elements.flClientRoundTable.innerHTML =
    tableRows.length === 0
      ? emptyTableRow(10, "client round 결과가 없습니다.")
      : tableRows
          .map(
            (row) => `
              <tr>
                <td>${escapeHtml(row.client_id)}</td>
                <td>${formatMetric(row.candidate_pseudo_label_count)}</td>
                <td>${formatMetric(row.accepted_pseudo_label_count)}</td>
                <td>${formatMetric(row.accepted_ratio)}</td>
                <td>${escapeHtml(row.update_status ?? "-")}</td>
                <td>${formatMetric(row.update_delta_l2)}</td>
                <td>${formatMetric(row.update_cosine_to_mean)}</td>
                <td>${formatBytes(row.payload_bytes)}</td>
                <td>${formatSeconds(row.train_seconds)}</td>
                <td>${formatMetric(row.pseudo_label_accuracy)}</td>
              </tr>
            `,
          )
          .join("");
}

function fillRunSelect(select, rows, selectedValue) {
  fillSelect(select, rows.map(runId), selectedValue, "run 없음");
  Array.from(select.options).forEach((option) => {
    const row = rows.find((candidate) => runId(candidate) === option.value);
    if (row) option.textContent = runDetailLabel(row);
  });
}

function fillRoundSelect(select, rows, selectedValue) {
  const indexes = Array.from(
    new Set(rows.map((row) => String(row.round_index)).filter((value) => value !== "null" && value !== "undefined")),
  ).sort((left, right) => Number(left) - Number(right));
  select.innerHTML = [
    `<option value="__latest__" ${selectedValue === "__latest__" ? "selected" : ""}>Latest</option>`,
    ...indexes.map((roundIndex) => {
      const selected = String(selectedValue) === roundIndex ? "selected" : "";
      return `<option value="${escapeHtml(roundIndex)}" ${selected}>round ${escapeHtml(roundIndex)}</option>`;
    }),
  ].join("");
}

function selectedClientRoundIndex(rows, state) {
  if (rows.length === 0) return null;
  if (state.clientRoundIndex !== "__latest__") return numberOrNull(state.clientRoundIndex);
  const indexes = rows.map((row) => numberOrNull(row.round_index)).filter((value) => value !== null);
  return indexes.length > 0 ? Math.max(...indexes) : null;
}

function normalizeRunId(selectedRunId, rows) {
  const runIds = new Set(rows.map(runId));
  if (selectedRunId && runIds.has(selectedRunId)) return selectedRunId;
  return rows.length > 0 ? runId(rows[0]) : null;
}
