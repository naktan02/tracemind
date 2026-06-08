import { formatMetric } from "../../../shared/formatting/numbers.js";
import { shortRun, shortSplit } from "../../../shared/formatting/text.js";

export function algorithmName(row) {
  return row.algorithm_name ?? row.method_name ?? "-";
}

export function peftAdapterLabel(row) {
  return row.peft_adapter_name ?? "-";
}

export function centralDataLabel(row) {
  const labeled = row.labeled_dataset_name ?? "?";
  const unlabeled = row.unlabeled_dataset_name ?? "?";
  return `${labeled} -> ${unlabeled}`;
}

export function overviewRunLabel(row) {
  return [
    algorithmName(row),
    `${peftAdapterLabel(row)} r${row.peft_adapter_rank ?? "?"}`,
    centralRunSuffix(row.run_id),
  ].join(" · ");
}

export function overviewDisplayLabel(row, aliases) {
  return aliases[row.run_id] || overviewRunLabel(row);
}

export function compareDisplayLabel(row, aliases) {
  return aliases[row.run_id] || algorithmName(row);
}

export function overviewRunSubLabel(row) {
  return [
    row.labeled_dataset_name ?? "?",
    "->",
    row.unlabeled_dataset_name ?? "?",
    `seed${row.seed ?? "?"}`,
  ].join(" ");
}

export function runDescriptor(row) {
  return [
    algorithmName(row),
    peftAdapterConfigLabel(row),
    `lr=${formatMetric(row.learning_rate)}`,
    `clf=${formatMetric(row.classifier_learning_rate)}`,
    shortSplit(row.selection_slug),
  ].join(" · ");
}

export function runDetail(row) {
  return [
    algorithmName(row),
    shortRun(row.run_id),
    runDescriptor(row),
    `labeled=${row.labeled_dataset_name ?? "-"}`,
    `unlabeled=${row.unlabeled_dataset_name ?? "-"}`,
    `validation=${row.validation_dataset_name ?? "-"}`,
    `test=${row.test_dataset_name ?? "-"}`,
    `run_id=${row.run_id}`,
  ].join(" · ");
}

export function peftAdapterConfigLabel(row) {
  return [
    `adapter=${peftAdapterLabel(row)}`,
    `r=${row.peft_adapter_rank ?? "-"}`,
    `alpha=${row.peft_adapter_alpha ?? "-"}`,
    `dropout=${row.peft_adapter_dropout ?? "-"}`,
  ].join(" · ");
}

export function centralRunSuffix(runId) {
  const match = String(runId ?? "").match(/(\d{4}_\d{2}_\d{2}_\d{6})$/);
  if (match) {
    return match[1].replace(/^2026_/, "");
  }
  return shortRun(runId);
}
