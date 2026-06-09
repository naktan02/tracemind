import {
  CENTRAL_INITIAL_EVAL_TRACK,
} from "./constants.js";
import { algorithmName } from "./labels.js";
import { compareMetricValues } from "../../../shared/formatting/metrics.js";

const CENTRAL_SSL_TRACK_PREFIX = "supervised";
const CENTRAL_TRACK_PREFIX = "central_";
const HIDDEN_CENTRAL_EVAL_SETS = new Set(["test"]);

function preferredCentralEvalSetOrder(evalSets) {
  const prioritized = [];
  for (const key of ["validation", "best", "final"]) {
    if (evalSets.delete(key)) {
      prioritized.push(key);
    }
  }
  return [...prioritized, ...Array.from(evalSets).sort()];
}

export function isCentralResultTrack(track) {
  const current = resolveTrack(track);
  if (!current.startsWith(CENTRAL_TRACK_PREFIX)) return false;
  return current !== CENTRAL_INITIAL_EVAL_TRACK;
}

export function isCentralSslResultTrack(track) {
  const current = resolveTrack(track);
  return isCentralResultTrack(current) && !isCentralSupervisedTrack(current);
}

export function isCentralSupervisedTrack(track) {
  const current = resolveTrack(track);
  if (!isCentralResultTrack(current)) {
    return false;
  }
  const methodName = resolveMethodName(track);
  const algorithm = resolveAlgorithmName(track);
  return current.includes(CENTRAL_SSL_TRACK_PREFIX) || methodName === "supervised" || algorithm === "supervised";
}

export function isFlResultTrack(track) {
  return resolveTrack(track).startsWith("fl_ssl");
}

export function isAllComparisonTrack(track) {
  const current = resolveTrack(track);
  return isCentralResultTrack(current) || isFlResultTrack(current);
}

export function centralEvalSets(bundle, trackPredicate = isCentralResultTrack) {
  const allCandidateSets = new Set(
    (bundle.eval_metrics ?? [])
      .filter((row) =>
        (bundle.runs ?? []).some(
          (run) => run.run_id === row.run_id && trackPredicate(run),
        ),
      )
      .map((row) => row.eval_set),
  );
  const candidateSets = new Set(allCandidateSets);
  for (const hidden of HIDDEN_CENTRAL_EVAL_SETS) {
    candidateSets.delete(hidden);
  }
  if (candidateSets.size === 0) {
    return preferredCentralEvalSetOrder(allCandidateSets);
  }
  return preferredCentralEvalSetOrder(candidateSets);
}

export function centralMetricRows(bundle, evalSet, sortMetric = "macro_f1", trackPredicate = isCentralResultTrack) {
  const runById = new Map((bundle.runs ?? []).map((run) => [run.run_id, run]));
  return (bundle.eval_metrics ?? [])
    .filter((row) => row.eval_set === evalSet)
    .map((row) => ({ ...runById.get(row.run_id), ...row }))
    .filter((row) => row.run_id && trackPredicate(row))
    .sort((left, right) =>
      compareMetricValues(left[sortMetric], right[sortMetric], sortMetric),
    );
}

export function centralAlgorithms(rows) {
  return Array.from(new Set(rows.map(algorithmName).filter(Boolean))).sort();
}

export function rowsForAlgorithms(rows, algorithms) {
  const selected = new Set(algorithms);
  return rows.filter((row) => selected.has(algorithmName(row)));
}

export function rowsWithProjection(bundle, rows, evalSet) {
  const projectionRunIds = new Set(
    (bundle.projection_images ?? [])
      .filter((image) => image.eval_set === evalSet)
      .map((image) => image.run_id),
  );
  return rows.filter((row) => projectionRunIds.has(row.run_id));
}

export function initialEvalRuns(bundle) {
  return (bundle.runs ?? []).filter((run) => run.track === CENTRAL_INITIAL_EVAL_TRACK);
}

function resolveTrack(value) {
  return String(value?.track ?? value ?? "");
}

function resolveMethodName(value) {
  return String(value?.method_name ?? "").toLowerCase();
}

function resolveAlgorithmName(value) {
  return String(value?.algorithm_name ?? "").toLowerCase();
}
