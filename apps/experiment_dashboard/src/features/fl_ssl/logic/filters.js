import { formatMetric } from "../../../shared/formatting/numbers.js";
import { applyFacetedFilters, optionsForAxis } from "../../../shared/filters/faceted_filters.js";
import {
  adapterKind,
  algorithmName,
  dataSourceLabel,
  labelBudgetLabel,
  localRegularizerLabel,
} from "./labels.js";
import { flRoundCountForRun } from "./selectors.js";

export function flFilterAxes(bundle) {
  return [
    axis("track", "Track", (row) => row.track ?? "-"),
    axis("algorithm", "Algorithm", algorithmName),
    axis("method_family", "Family", (row) => row.method_family ?? "-"),
    axis("local_regularizer", "Regularizer", localRegularizerLabel),
    axis("peft_adapter", "Adapter", (row) => row.peft_adapter_name ?? "-"),
    axis("peft_adapter_rank", "Adapter Rank", (row) => row.peft_adapter_rank ?? "-", (row) => `rank ${row.peft_adapter_rank ?? "-"}`),
    axis("data_pair", "Labeled / Unlabeled", dataSourceLabel),
    axis("label_budget", "Label Budget", labelBudgetLabel),
    axis("round_count", "Round Count", (row) => flRoundCountForRun(bundle, row) ?? "-", (row) => `${flRoundCountForRun(bundle, row) ?? "-"} rounds`),
    axis("local_epochs", "Local Epochs", (row) => row.epochs ?? "-", (row) => `${row.epochs ?? "-"} local epochs`),
    axis("client_count", "Client Count", (row) => row.client_count ?? "-", (row) => `${row.client_count ?? "-"} clients`),
    axis("adapter", "Adapter Payload", adapterKind),
    axis("aggregation", "Aggregation", (row) => row.aggregation_backend_name ?? "-"),
    axis("seed", "Seed", (row) => row.seed ?? "-", (row) => `seed ${row.seed ?? "-"}`),
    axis("shard_alpha", "Shard Alpha", (row) => row.shard_alpha ?? "-", (row) => `alpha ${formatMetric(row.shard_alpha)}`),
    axis("learning_rate", "Learning Rate", (row) => row.learning_rate ?? "-", (row) => formatMetric(row.learning_rate)),
    axis("classifier_learning_rate", "Classifier LR", (row) => row.classifier_learning_rate ?? "-", (row) => formatMetric(row.classifier_learning_rate)),
    axis("max_train_steps", "Max Steps", (row) => row.max_train_steps ?? "-", (row) => `${row.max_train_steps ?? "-"} steps`),
    axis("train_batch_size", "Train Batch", (row) => row.train_batch_size ?? "-", (row) => `batch ${row.train_batch_size ?? "-"}`),
  ];
}

export function applyFlFilters(bundle, rows, flState) {
  return applyFacetedFilters(rows, flFilterAxes(bundle), flState.filterAxisIds, flState.filterValues);
}

export function pruneFlFilters(bundle, rows, flState) {
  const axes = flFilterAxes(bundle);
  const visibleAxisIds = new Set(
    axes
      .filter(
        (axisDef) =>
          optionsForAxis(rows, axisDef, axes, flState.filterAxisIds, flState.filterValues)
            .length > 1,
      )
      .map((axisDef) => axisDef.id),
  );
  flState.filterAxisIds = flState.filterAxisIds.filter((axisId) =>
    visibleAxisIds.has(axisId),
  );
  for (const axisId of Object.keys(flState.filterValues)) {
    if (!flState.filterAxisIds.includes(axisId)) {
      delete flState.filterValues[axisId];
      continue;
    }
    const axisDef = axes.find((candidate) => candidate.id === axisId);
    const validValues = new Set(
      axisDef
        ? optionsForAxis(rows, axisDef, axes, flState.filterAxisIds, flState.filterValues).map(
            (option) => option.value,
          )
        : [],
    );
    flState.filterValues[axisId] = (flState.filterValues[axisId] ?? []).filter(
      (value) => validValues.has(value),
    );
    if (flState.filterValues[axisId].length === 0) {
      delete flState.filterValues[axisId];
    }
  }
}

function axis(id, label, value, labelForValue = null) {
  return {
    id,
    label,
    value: (row) => String(value(row)),
    labelForValue: labelForValue ? (row) => labelForValue(row) : null,
  };
}
