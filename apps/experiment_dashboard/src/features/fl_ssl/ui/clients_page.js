import { escapeHtml } from "../../../shared/formatting/html.js";
import { formatBytes, formatMetric, formatSeconds, numberOrNull } from "../../../shared/formatting/numbers.js";
import { fillSelect } from "../../../ui/controls/form_controls.js";
import {
  emptyTableRow,
  moveTableColumn,
  renderSortableTableHeader,
  resolveTableColumns,
} from "../../../ui/tables/table.js";
import { runDetailLabel, runId } from "../logic/labels.js";
import {
  compareClientRoundRows,
  flClientRoundRows,
  flClientValidationRows,
  flRunsWithRows,
} from "../logic/selectors.js";

const CLIENT_VALIDATION_COLUMNS = [
  { id: "client", label: "client", group: "axis", render: (row) => escapeHtml(row.client_id) },
  {
    id: "labeled",
    label: "labeled",
    group: "metric",
    render: (row) => formatMetric(valueFor(row, ["labeled_count", "client_labeled_count"])),
  },
  {
    id: "unlabeled",
    label: "unlabeled",
    group: "metric",
    render: (row) => formatMetric(valueFor(row, ["unlabeled_count", "client_unlabeled_count"])),
  },
  {
    id: "macro_f1",
    label: "macro_f1",
    group: "metric",
    render: (row) => formatMetric(valueFor(row, ["macro_f1", "client_validation_macro_f1"])),
  },
  {
    id: "accuracy",
    label: "accuracy",
    group: "metric",
    render: (row) =>
      formatMetric(valueFor(row, ["accuracy_top_1", "client_validation_accuracy_top_1"])),
  },
  {
    id: "loss",
    label: "loss",
    group: "metric",
    render: (row) => formatMetric(valueFor(row, ["loss", "client_validation_loss"])),
  },
  {
    id: "ece",
    label: "ece",
    group: "metric",
    render: (row) =>
      formatMetric(
        valueFor(row, [
          "expected_calibration_error",
          "ece",
          "client_validation_ece",
        ]),
      ),
  },
  {
    id: "accepted",
    label: "accepted",
    group: "metric",
    render: (row) =>
      formatMetric(
        valueFor(row, [
          "total_accepted_pseudo_labels",
          "accepted_pseudo_label_count",
          "client_accepted_count",
        ]),
      ),
  },
  {
    id: "update_rounds",
    label: "update rounds",
    group: "metric",
    render: (row) =>
      formatMetric(valueFor(row, ["update_round_count", "update_generated_round_count"])),
  },
  {
    id: "delta_norm",
    label: "delta norm",
    group: "metric",
    render: (row) => formatMetric(valueFor(row, ["mean_update_delta_l2", "mean_delta_l2_norm"])),
  },
];

const CLIENT_ROUND_COLUMNS = [
  { id: "client", label: "client", group: "axis", render: (row) => escapeHtml(row.client_id) },
  {
    id: "candidate",
    label: "candidate",
    group: "metric",
    render: (row) =>
      formatMetric(
        valueFor(
          row,
          ["candidate_pseudo_label_count", "candidate_count", "client_candidate_count"],
        ),
      ),
  },
  {
    id: "accepted",
    label: "accepted",
    group: "metric",
    render: (row) =>
      formatMetric(
        valueFor(
          row,
          ["accepted_pseudo_label_count", "accepted_count", "client_accepted_count"],
        ),
      ),
  },
  {
    id: "ratio",
    label: "ratio",
    group: "metric",
    render: (row) =>
      formatMetric(valueFor(row, ["accepted_ratio", "client_accepted_ratio", "accepted_pseudo_ratio"])),
  },
  {
    id: "update",
    label: "update",
    group: "metric",
    render: (row) =>
      formatMetric(
        valueFor(
          row,
          ["update_status", "update_generated", "client_update_generated"],
        ),
      ),
  },
  {
    id: "delta_norm",
    label: "delta norm",
    group: "metric",
    render: (row) => formatMetric(valueFor(row, ["update_delta_l2", "delta_l2_norm"])),
  },
  {
    id: "cos_mean",
    label: "cos mean",
    group: "metric",
    render: (row) =>
      formatMetric(valueFor(row, ["update_cosine_to_mean", "per_client_delta_cosine_to_mean", "cosine_to_mean"])),
  },
  {
    id: "payload",
    label: "payload",
    group: "metric",
    render: (row) => formatBytes(valueFor(row, ["payload_bytes", "client_payload_bytes"])),
  },
  {
    id: "seconds",
    label: "seconds",
    group: "metric",
    render: (row) => formatSeconds(valueFor(row, ["train_seconds", "client_train_time_seconds"])),
  },
  {
    id: "pseudo_acc",
    label: "pseudo acc",
    group: "metric",
    render: (row) => formatMetric(row.pseudo_label_accuracy),
  },
];

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

export function renderClientsPage(elements, rows, state, bundle, rerender = () => {}) {
  renderClientValidation(elements, rows, state, bundle, rerender);
  renderClientRounds(elements, rows, state, bundle, rerender);
}

function valueFor(row, keys, defaultValue = null) {
  for (const key of keys) {
    const value = row[key];
    if (value !== undefined && value !== null) return value;
  }
  return defaultValue;
}

function renderClientValidation(elements, rows, state, bundle, rerender) {
  const candidateRuns = flRunsWithRows(rows, flClientValidationRows(bundle));
  fillRunSelect(elements.flClientValidationRunFilter, candidateRuns, state.clientValidationRunId);
  const tableRows = flClientValidationRows(bundle)
    .filter((row) => row.run_id === state.clientValidationRunId)
    .sort((left, right) => String(left.client_id ?? "").localeCompare(String(right.client_id ?? "")));

  const { visibleColumns } = resolveTableColumns(
    state.clientValidationTableColumns,
    CLIENT_VALIDATION_COLUMNS,
    CLIENT_VALIDATION_COLUMNS.map((column) => column.id),
  );
  renderSortableTableHeader(
    elements.flClientValidationTableHead,
    visibleColumns,
    (sourceColumnId, targetColumnId) => {
      if (moveTableColumn(state.clientValidationTableColumns, sourceColumnId, targetColumnId)) {
        rerender();
      }
    },
  );
  elements.flClientValidationTable.innerHTML =
    tableRows.length === 0
      ? emptyTableRow(visibleColumns.length || 1, "client validation 결과가 없습니다.")
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

function renderClientRounds(elements, rows, state, bundle, rerender) {
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

  const { visibleColumns } = resolveTableColumns(
    state.clientRoundTableColumns,
    CLIENT_ROUND_COLUMNS,
    CLIENT_ROUND_COLUMNS.map((column) => column.id),
  );
  renderSortableTableHeader(
    elements.flClientRoundTableHead,
    visibleColumns,
    (sourceColumnId, targetColumnId) => {
      if (moveTableColumn(state.clientRoundTableColumns, sourceColumnId, targetColumnId)) {
        rerender();
      }
    },
  );
  elements.flClientRoundTable.innerHTML =
    tableRows.length === 0
      ? emptyTableRow(visibleColumns.length || 1, "client round 결과가 없습니다.")
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
