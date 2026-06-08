import { escapeHtml } from "../../../shared/formatting/html.js";
import { formatBytes, formatMetric, formatSeconds, numberOrNull } from "../../../shared/formatting/numbers.js";
import { fillSelect } from "../../../ui/controls/form_controls.js";
import { emptyTableRow } from "../../../ui/tables/table.js";
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

function valueFor(row, keys, defaultValue = null) {
  for (const key of keys) {
    const value = row[key];
    if (value !== undefined && value !== null) return value;
  }
  return defaultValue;
}

function formatClientStatus(value) {
  if (value === null || value === undefined) return "-";
  if (value === true) return "true";
  if (value === false) return "false";
  return String(value);
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
                <td>${formatMetric(
                  valueFor(row, ["labeled_count", "client_labeled_count"]),
                )}</td>
                <td>${formatMetric(
                  valueFor(row, ["unlabeled_count", "client_unlabeled_count"]),
                )}</td>
                <td>${formatMetric(
                  valueFor(row, ["macro_f1", "client_validation_macro_f1"]),
                )}</td>
                <td>${formatMetric(
                  valueFor(row, ["accuracy_top_1", "client_validation_accuracy_top_1"]),
                )}</td>
                <td>${formatMetric(
                  valueFor(row, ["loss", "client_validation_loss"]),
                )}</td>
                <td>${formatMetric(
                  valueFor(row, [
                    "expected_calibration_error",
                    "ece",
                    "client_validation_ece",
                  ]),
                )}</td>
                <td>${formatMetric(
                  valueFor(
                    row,
                    [
                      "total_accepted_pseudo_labels",
                      "accepted_pseudo_label_count",
                      "client_accepted_count",
                    ],
                  ),
                )}</td>
                <td>${formatMetric(
                  valueFor(row, ["update_round_count", "update_generated_round_count"]),
                )}</td>
                <td>${formatMetric(
                  valueFor(row, ["mean_update_delta_l2", "mean_delta_l2_norm"]),
                )}</td>
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
                <td>${formatMetric(
                  valueFor(
                    row,
                    [
                      "candidate_pseudo_label_count",
                      "candidate_count",
                      "client_candidate_count",
                    ],
                  ),
                )}</td>
                <td>${formatMetric(
                  valueFor(
                    row,
                    [
                      "accepted_pseudo_label_count",
                      "accepted_count",
                      "client_accepted_count",
                    ],
                  ),
                )}</td>
                <td>${formatMetric(
                  valueFor(
                    row,
                    [
                      "accepted_ratio",
                      "client_accepted_ratio",
                      "accepted_pseudo_ratio",
                    ],
                  ),
                )}</td>
                <td>${escapeHtml(
                  formatClientStatus(
                    valueFor(
                      row,
                      ["update_status", "update_generated", "client_update_generated"],
                    ),
                  ),
                )}</td>
                <td>${formatMetric(valueFor(row, ["update_delta_l2", "delta_l2_norm"]))}</td>
                <td>${formatMetric(
                  valueFor(
                    row,
                    [
                      "update_cosine_to_mean",
                      "per_client_delta_cosine_to_mean",
                      "cosine_to_mean",
                    ],
                  ),
                )}</td>
                <td>${formatBytes(valueFor(row, ["payload_bytes", "client_payload_bytes"]))}</td>
                <td>${formatSeconds(valueFor(row, ["train_seconds", "client_train_time_seconds"]))}</td>
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
