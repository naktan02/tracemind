import {
  CENTRAL_INITIAL_EVAL_TRACK,
  CENTRAL_SSL_TRACK,
} from "./constants.js";
import { algorithmName } from "./labels.js";
import { compareMetricValues } from "../../../shared/formatting/metrics.js";

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
  return track === CENTRAL_SSL_TRACK || track === "central_peft_full_encoder_ssl";
}

export function centralEvalSets(bundle) {
  const candidateSets = new Set(
    (bundle.eval_metrics ?? [])
      .filter((row) =>
        (bundle.runs ?? []).some(
          (run) => run.run_id === row.run_id && isCentralResultTrack(run.track),
        ),
      )
      .map((row) => row.eval_set),
  );
  for (const hidden of HIDDEN_CENTRAL_EVAL_SETS) {
    candidateSets.delete(hidden);
  }
  return preferredCentralEvalSetOrder(candidateSets);
}

export function centralMetricRows(bundle, evalSet, sortMetric = "macro_f1") {
  const runById = new Map((bundle.runs ?? []).map((run) => [run.run_id, run]));
  return (bundle.eval_metrics ?? [])
    .filter((row) => row.eval_set === evalSet)
    .map((row) => ({ ...runById.get(row.run_id), ...row }))
    .filter((row) => row.run_id && isCentralResultTrack(row.track))
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
