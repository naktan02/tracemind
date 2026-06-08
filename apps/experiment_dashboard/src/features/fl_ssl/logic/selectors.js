import { numberOrNull } from "../../../shared/formatting/numbers.js";
import { compareMetricValues } from "../../../shared/formatting/metrics.js";
import { runId } from "./labels.js";

export function isFlSslTrack(track) {
  return String(track ?? "").startsWith("fl_ssl");
}

export function flSslRows(bundle) {
  return Array.isArray(bundle.fl_ssl_runs)
    ? bundle.fl_ssl_runs
    : (bundle.runs ?? []).filter((run) => isFlSslTrack(run.track));
}

export function flRoundRows(bundle) {
  return Array.isArray(bundle.fl_ssl_rounds) ? bundle.fl_ssl_rounds : [];
}

export function flClientRoundRows(bundle) {
  return Array.isArray(bundle.fl_ssl_client_rounds)
    ? bundle.fl_ssl_client_rounds
    : [];
}

export function flClientValidationRows(bundle) {
  return Array.isArray(bundle.fl_ssl_client_validations)
    ? bundle.fl_ssl_client_validations
    : [];
}

export function flSplitRows(bundle) {
  return Array.isArray(bundle.fl_ssl_client_splits)
    ? bundle.fl_ssl_client_splits
    : [];
}

export function sortedFlRows(rows) {
  return rows.slice().sort((left, right) => compareFlRows(left, right, "macro_f1"));
}

export function flRunsWithRows(runs, dataRows) {
  const runIds = new Set(dataRows.map((row) => row.run_id));
  return runs.filter((row) => runIds.has(runId(row)));
}

export function flRoundRowsForRun(bundle, selectedRunId) {
  return flRoundRows(bundle)
    .filter((row) => row.run_id === selectedRunId)
    .sort(compareRoundRows);
}

export function flRoundCountForRun(bundle, row) {
  const completedRounds = numberOrNull(row.completed_rounds);
  if (completedRounds !== null) return completedRounds;
  const indexes = flRoundRowsForRun(bundle, runId(row))
    .map((roundRow) => numberOrNull(roundRow.round_index))
    .filter((roundIndex) => roundIndex !== null);
  return indexes.length > 0 ? Math.max(...indexes) : null;
}

export function flProjectionEvalSets(bundle, rows) {
  const runIds = new Set(rows.map(runId));
  return Array.from(
    new Set(
      (bundle.projection_images ?? [])
        .filter((image) => runIds.has(image.run_id))
        .map((image) => image.eval_set),
    ),
  ).sort();
}

export function flRowsWithProjection(bundle, rows, evalSet) {
  const projectionRunIds = new Set(
    (bundle.projection_images ?? [])
      .filter((image) => image.eval_set === evalSet)
      .map((image) => image.run_id),
  );
  return rows.filter((row) => projectionRunIds.has(runId(row)));
}

export function compareRoundRows(left, right) {
  return compareNullableNumbers(left.round_index, right.round_index);
}

export function compareFlRoundRows(left, right) {
  const runCompare = String(left.run_id ?? "").localeCompare(String(right.run_id ?? ""));
  return runCompare !== 0 ? runCompare : compareRoundRows(left, right);
}

export function compareClientRoundRows(left, right) {
  const roundCompare = compareRoundRows(left, right);
  return roundCompare !== 0
    ? roundCompare
    : String(left.client_id ?? "").localeCompare(String(right.client_id ?? ""));
}

function compareFlRows(left, right, metric) {
  return compareMetricValues(flMetricValue(left, metric), flMetricValue(right, metric), metric);
}

function flMetricValue(row, metric) {
  if (row[metric] !== undefined) return row[metric];
  if (row.metrics?.primary?.[metric] !== undefined) return row.metrics.primary[metric];
  if (row.metrics?.secondary?.[metric] !== undefined) return row.metrics.secondary[metric];
  return null;
}

function compareNullableNumbers(leftValue, rightValue) {
  const left = numberOrNull(leftValue);
  const right = numberOrNull(rightValue);
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return left - right;
}
