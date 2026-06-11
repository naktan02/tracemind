import { CENTRAL_INITIAL_METRIC_MAP } from "./constants.js";
import { initialEvalRuns } from "./selectors.js";
import { numberOrNull } from "../../../shared/formatting/numbers.js";

export function centralInitialPoint(bundle, row, metric, evalSet) {
  const value = centralInitialMetricValue(bundle, row, metric, evalSet);
  return value === null ? null : { step: 0, epoch: null, value };
}

export function centralInitialMetricValue(bundle, row, metric, evalSet) {
  const initialRun = centralInitialRunFor(bundle, row);
  if (!initialRun) return null;
  const evalSetCandidates = buildInitialEvalSetCandidates(evalSet);
  const evalMetric = evalSetCandidates
    .map((candidateEvalSet) =>
      (bundle.eval_metrics ?? []).find(
        (candidate) =>
          candidate.run_id === initialRun.run_id && candidate.eval_set === candidateEvalSet,
      ),
    )
    .find(Boolean);
  if (!evalMetric) return null;
  const initialMetric = CENTRAL_INITIAL_METRIC_MAP[metric];
  return initialMetric ? numberOrNull(evalMetric[initialMetric]) : null;
}

function buildInitialEvalSetCandidates(evalSet) {
  const candidates = [evalSet, "validation", "test", "final_validation", "initial_validation"];
  return Array.from(new Set(candidates.filter((candidate) => Boolean(candidate))));
}

export function centralInitialRunFor(bundle, row) {
  return initialEvalRuns(bundle).find((candidate) => {
    return (
      candidate.seed === row.seed &&
      candidate.labeled_dataset_name === row.labeled_dataset_name &&
      candidate.unlabeled_dataset_name === row.unlabeled_dataset_name &&
      candidate.test_dataset_name === row.test_dataset_name &&
      compatibleValidationDataset(candidate, row) &&
      compatibleAdapterConfig(candidate, row)
    );
  });
}

function compatibleValidationDataset(candidate, row) {
  if (!candidate.validation_dataset_name || !row.validation_dataset_name) {
    return true;
  }
  return candidate.validation_dataset_name === row.validation_dataset_name;
}

function compatibleAdapterConfig(candidate, row) {
  if (!hasAdapterConfig(candidate) || !hasAdapterConfig(row)) {
    return true;
  }
  return (
    candidate.peft_adapter_name === row.peft_adapter_name &&
    candidate.peft_adapter_rank === row.peft_adapter_rank &&
    candidate.peft_adapter_alpha === row.peft_adapter_alpha &&
    candidate.peft_adapter_dropout === row.peft_adapter_dropout
  );
}

function hasAdapterConfig(row) {
  return (
    row.peft_adapter_name !== null &&
    row.peft_adapter_name !== undefined &&
    row.peft_adapter_rank !== null &&
    row.peft_adapter_rank !== undefined
  );
}
