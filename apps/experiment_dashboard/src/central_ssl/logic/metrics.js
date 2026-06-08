import {
  CENTRAL_EPOCH_METRICS,
  CENTRAL_OVERVIEW_METRICS,
} from "./constants.js";
import { numberOrNull } from "../../shared/formatting/numbers.js";
import { uniqueValues } from "../../shared/formatting/text.js";

const NON_DISPLAY_METRICS = new Set([
  "seed",
  "learning_rate",
  "classifier_learning_rate",
  "epochs",
  "max_train_steps",
  "train_batch_size",
  "eval_batch_size",
  "peft_adapter_rank",
  "peft_adapter_alpha",
  "peft_adapter_dropout",
  "created_at",
]);

export function centralOverviewMetricKeys(rows) {
  const discovered = new Set();
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) {
      if (!NON_DISPLAY_METRICS.has(key) && typeof value !== "boolean" && numberOrNull(value) !== null) {
        discovered.add(key);
      }
    }
  }
  return uniqueValues([
    ...CENTRAL_OVERVIEW_METRICS.filter((metric) => discovered.has(metric)),
    ...Array.from(discovered).sort(),
  ]);
}

export function centralEpochMetricKeys(bundle) {
  const discovered = new Set();
  for (const row of bundle.epoch_metrics ?? []) {
    for (const key of Object.keys(row)) {
      if (!["run_id", "epoch", "step"].includes(key) && numberOrNull(row[key]) !== null) {
        discovered.add(key);
      }
    }
  }
  return uniqueValues([
    ...CENTRAL_EPOCH_METRICS.filter((metric) => discovered.has(metric)),
    ...Array.from(discovered).sort(),
  ]);
}

export function centralLatestMetricValue(bundle, runId, metric) {
  const points = centralEpochPoints(bundle, runId, metric);
  return points.length > 0 ? points[points.length - 1].value : null;
}

export function centralEpochPoints(bundle, runId, metric) {
  return (bundle.epoch_metrics ?? [])
    .filter((row) => row.run_id === runId)
    .map((row, index) => ({
      step: numberOrNull(row.step) ?? numberOrNull(row.epoch) ?? index + 1,
      epoch: numberOrNull(row.epoch),
      value: numberOrNull(row[metric]),
    }))
    .filter((point) => point.value !== null)
    .sort((left, right) => left.step - right.step);
}

export function formatStepTick(step) {
  return String(step);
}
