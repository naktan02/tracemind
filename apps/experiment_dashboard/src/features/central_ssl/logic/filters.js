import { formatMetric } from "../../../shared/formatting/numbers.js";
import { applyFacetedFilters, optionsForAxis } from "../../../shared/filters/faceted_filters.js";
import {
  algorithmName,
  centralDataLabel,
  evaluationDataLabel,
  initialCheckpointLabel,
  isCentralSupervisedRow,
  labelBudgetLabel,
  peftAdapterLabel,
  runCreatedDateLabel,
  trainingDataLabel,
} from "./labels.js";

export const CENTRAL_FILTER_AXES = [
  axis("track", "Track", (row) => row.track ?? "-"),
  axis("algorithm", "Algorithm", algorithmName),
  axis("method_family", "Family", (row) => row.method_family ?? "-"),
  axis("training_data", "Training Data", trainingDataLabel),
  axis("evaluation_data", "Evaluation Data", evaluationDataLabel),
  axis("label_budget", "Label Budget", labelBudgetLabel),
  axis("data_pair", "Labeled / Unlabeled", centralDataLabel),
  axis("labeled_dataset", "Labeled Dataset", (row) => row.labeled_dataset_name ?? "-"),
  axis("unlabeled_dataset", "Unlabeled Dataset", (row) =>
    isCentralSupervisedRow(row) ? "-" : row.unlabeled_dataset_name ?? "-",
  ),
  axis("validation_dataset", "Validation Dataset", (row) =>
    isCentralSupervisedRow(row) ? "-" : row.validation_dataset_name ?? "-",
  ),
  axis("test_dataset", "Test Dataset", (row) => row.test_dataset_name ?? "-"),
  axis("initial_checkpoint", "Initial Checkpoint", initialCheckpointLabel, null, {
    alwaysVisible: true,
  }),
  axis("created_date", "Run Date", runCreatedDateLabel, null, {
    alwaysVisible: true,
  }),
  axis("peft_adapter", "Adapter", peftAdapterLabel),
  axis("peft_adapter_rank", "Adapter Rank", (row) => row.peft_adapter_rank ?? "-", (row) => `rank ${row.peft_adapter_rank ?? "-"}`),
  axis("peft_adapter_alpha", "Adapter Alpha", (row) => row.peft_adapter_alpha ?? "-", (row) => `alpha ${row.peft_adapter_alpha ?? "-"}`),
  axis("seed", "Seed", (row) => row.seed ?? "-", (row) => `seed ${row.seed ?? "-"}`),
  axis("learning_rate", "Learning Rate", (row) => row.learning_rate ?? "-", (row) => formatMetric(row.learning_rate)),
  axis("classifier_learning_rate", "Classifier LR", (row) => row.classifier_learning_rate ?? "-", (row) => formatMetric(row.classifier_learning_rate)),
  axis("epochs", "Epochs", (row) => row.epochs ?? "-", (row) => `${row.epochs ?? "-"} epochs`),
  axis("max_train_steps", "Max Steps", (row) => row.max_train_steps ?? "-", (row) => `${row.max_train_steps ?? "-"} steps`),
  axis("train_batch_size", "Train Batch", (row) => row.train_batch_size ?? "-", (row) => `batch ${row.train_batch_size ?? "-"}`),
];

export function applyCentralFilters(rows, centralState) {
  return applyFacetedFilters(
    rows,
    CENTRAL_FILTER_AXES,
    centralState.filterAxisIds,
    centralState.filterValues,
  );
}

export function pruneCentralFilters(rows, centralState) {
  const visibleAxisIds = new Set(
    CENTRAL_FILTER_AXES.filter(
      (axis) => {
        const optionCount = optionsForAxis(
          rows,
          axis,
          CENTRAL_FILTER_AXES,
          centralState.filterAxisIds,
          centralState.filterValues,
        ).length;
        return optionCount > 1 || (axis.alwaysVisible && optionCount > 0);
      },
    ).map((axis) => axis.id),
  );
  centralState.filterAxisIds = centralState.filterAxisIds.filter((axisId) =>
    visibleAxisIds.has(axisId),
  );
  for (const axisId of Object.keys(centralState.filterValues)) {
    if (!centralState.filterAxisIds.includes(axisId)) {
      delete centralState.filterValues[axisId];
      continue;
    }
    const axis = CENTRAL_FILTER_AXES.find((candidate) => candidate.id === axisId);
    const validValues = new Set(
      axis
        ? optionsForAxis(
            rows,
            axis,
            CENTRAL_FILTER_AXES,
            centralState.filterAxisIds,
            centralState.filterValues,
          ).map((option) => option.value)
        : [],
    );
    centralState.filterValues[axisId] = (centralState.filterValues[axisId] ?? []).filter(
      (value) => validValues.has(value),
    );
    if (centralState.filterValues[axisId].length === 0) {
      delete centralState.filterValues[axisId];
    }
  }
}

function axis(id, label, value, labelForValue = null, options = {}) {
  return {
    id,
    label,
    value: (row) => String(value(row)),
    labelForValue: labelForValue ? (row) => labelForValue(row) : null,
    alwaysVisible: Boolean(options.alwaysVisible),
  };
}
