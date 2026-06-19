import { formatMetric } from "../../../shared/formatting/numbers.js";
import {
  compactDate,
  compactDateTime,
  shortRun,
  shortSplit,
} from "../../../shared/formatting/text.js";

export function algorithmName(row) {
  const methodName = row.method_name ?? "-";
  if (methodName === "supervised" && row.method_family) {
    return `supervised · ${methodFamilyLabel(row)}`;
  }
  return row.algorithm_name ?? methodName;
}

export function peftAdapterLabel(row) {
  return row.peft_adapter_name ?? "-";
}

export function methodFamilyLabel(row) {
  if (row.method_family === "peft_classifier") return "PEFT classifier";
  if (row.method_family === "full_text_encoder") return "full text encoder";
  return row.method_family ?? "-";
}

export function isCentralSupervisedRow(row) {
  const track = String(row.track ?? "");
  return track.includes("supervised") || row.method_name === "supervised";
}

export function labelBudgetLabel(row) {
  if (row.label_budget_name) return row.label_budget_name;
  if (row.label_budget_count_per_class) {
    return `pc${row.label_budget_count_per_class}`;
  }
  const source = [row.selection_slug, row.run_id].join(" ");
  const labelsPcMatch = source.match(/labels_pc(\d+)/);
  if (labelsPcMatch) return `pc${labelsPcMatch[1]}`;
  const labeledPerClassMatch = source.match(/labeled(\d+)_per_class/);
  if (labeledPerClassMatch) return `pc${labeledPerClassMatch[1]}`;
  return "pc?";
}

export function centralDataLabel(row) {
  const labeled = row.labeled_dataset_name ?? "?";
  const unlabeled = row.unlabeled_dataset_name ?? "?";
  if (isCentralSupervisedRow(row)) {
    return `L:${labeled} ${labelBudgetLabel(row)}`;
  }
  return `${labeled} -> ${unlabeled}`;
}

export function trainingDataLabel(row) {
  const labeled = row.labeled_dataset_name ?? "?";
  if (isCentralSupervisedRow(row)) {
    return `labeled=${labeled} ${labelBudgetLabel(row)}`;
  }
  return [
    `labeled=${labeled} ${labelBudgetLabel(row)}`,
    `unlabeled=${row.unlabeled_dataset_name ?? "?"}`,
  ].join(" · ");
}

export function evaluationDataLabel(row) {
  if (isCentralSupervisedRow(row)) {
    return `test=${row.test_dataset_name ?? "?"}`;
  }
  const validation = row.validation_dataset_name;
  const test = row.test_dataset_name ?? "?";
  return [
    validation && validation !== test ? `validation=${validation}` : null,
    `test=${test}`,
  ].filter(Boolean).join(" · ");
}

export function initialCheckpointLabel(row) {
  const raw = String(row.initial_checkpoint_name ?? "").trim();
  if (isCentralSupervisedRow(row)) {
    return raw || "-";
  }
  return raw || "unrecorded";
}

export function backboneModelLabel(row) {
  return row.backbone_model_id ?? row.embedding_model_id ?? "-";
}

export function runCreatedDateLabel(row) {
  return compactDate(row.created_at);
}

export function overviewRunLabel(row) {
  const modelConfig = modelConfigLabel(row);
  return [
    algorithmName(row),
    modelConfig === methodFamilyLabel(row) ? null : modelConfig,
    compactDateTime(row.created_at) !== "-"
      ? compactDateTime(row.created_at)
      : centralRunSuffix(row.run_id),
  ].filter(Boolean).join(" · ");
}

export function overviewDisplayLabel(row, aliases) {
  return aliases[row.run_id] || overviewRunLabel(row);
}

export function compareDisplayLabel(row, aliases) {
  return aliases[row.run_id] || algorithmName(row);
}

export function overviewRunSubLabel(row) {
  return [
    trainingDataLabel(row),
    evaluationDataLabel(row),
    `model=${backboneModelLabel(row)}`,
    `ckpt=${initialCheckpointLabel(row)}`,
    `seed=${row.seed ?? "?"}`,
  ].join(" · ");
}

export function runDescriptor(row) {
  return [
    `family=${methodFamilyLabel(row)}`,
    `model=${backboneModelLabel(row)}`,
    trainingDataLabel(row),
    evaluationDataLabel(row),
    peftAdapterConfigLabel(row),
    batchConfigLabel(row),
    `lr=${formatHyperparameter(row.learning_rate)}`,
    `clf=${formatHyperparameter(row.classifier_learning_rate)}`,
  ].join(" · ");
}

export function runDetail(row) {
  const parts = [
    algorithmName(row),
    shortRun(row.run_id),
    runDescriptor(row),
    `label_budget=${labelBudgetLabel(row)}`,
    `checkpoint=${initialCheckpointLabel(row)}`,
    `created=${compactDateTime(row.created_at)}`,
    `run_id=${row.run_id}`,
  ];
  if (!isCentralSupervisedRow(row)) {
    parts.push(`split=${shortSplit(row.selection_slug)}`);
  }
  return parts.join(" · ");
}

export function centralEvalSetLabel(evalSet) {
  if (evalSet === "best") return "best_checkpoint";
  if (evalSet === "final") return "final_epoch";
  return evalSet;
}

export function peftAdapterConfigLabel(row) {
  if (!row.peft_adapter_name && !row.peft_adapter_rank) {
    return `surface=${methodFamilyLabel(row)}`;
  }
  return [
    `adapter=${peftAdapterLabel(row)}`,
    `r=${row.peft_adapter_rank ?? "-"}`,
    `alpha=${row.peft_adapter_alpha ?? "-"}`,
    `dropout=${formatHyperparameter(row.peft_adapter_dropout)}`,
  ].join(" · ");
}

function formatHyperparameter(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (number !== 0 && Math.abs(number) < 0.0001) {
    return number.toExponential(1);
  }
  return formatMetric(number);
}

function modelConfigLabel(row) {
  if (!row.peft_adapter_name && !row.peft_adapter_rank) {
    return methodFamilyLabel(row);
  }
  return `${peftAdapterLabel(row)} r${row.peft_adapter_rank ?? "?"}`;
}

function batchConfigLabel(row) {
  const labeled = row.labeled_batch_size ?? row.train_batch_size ?? "-";
  if (isCentralSupervisedRow(row)) {
    return `labeled_batch=${labeled}`;
  }
  return [
    `labeled_batch=${labeled}`,
    `unlabeled_batch=${row.unlabeled_batch_size ?? "-"}`,
  ].join(" · ");
}

export function centralRunSuffix(runId) {
  const match = String(runId ?? "").match(/(\d{4}_\d{2}_\d{2}_\d{6})$/);
  if (match) {
    return match[1].replace(/^2026_/, "");
  }
  return shortRun(runId);
}
