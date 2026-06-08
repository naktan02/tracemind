const METRIC_LABELS = {
  accuracy_top_1: "accuracy",
  expected_calibration_error: "ECE",
  max_calibration_error: "max ECE",
  macro_f1: "macro F1",
  weighted_f1: "weighted F1",
  worst_category_f1_value: "worst class F1",
  worst_client_macro_f1: "worst client F1",
  best_client_macro_f1: "best client F1",
  macro_f1_std: "macro F1 std",
  final_macro_f1: "final macro F1",
  final_accuracy_top_1: "final accuracy",
  final_loss: "final loss",
  final_expected_calibration_error: "final ECE",
  selection_macro_f1: "selection macro F1",
  selection_accuracy_top_1: "selection accuracy",
  selection_expected_calibration_error: "selection ECE",
  selection_worst_category_f1_value: "selection worst F1",
  selection_loss: "selection loss",
  train_loss: "train loss",
  train_sup_loss: "supervised loss",
  train_unsup_loss: "unsupervised loss",
};

export function metricLabel(metric) {
  return METRIC_LABELS[metric] ?? String(metric ?? "-").replace(/_/g, " ");
}

export function compareMetricValues(leftValue, rightValue, metric) {
  const left = Number(leftValue);
  const right = Number(rightValue);
  if (!Number.isFinite(left) && !Number.isFinite(right)) return 0;
  if (!Number.isFinite(left)) return 1;
  if (!Number.isFinite(right)) return -1;
  return metric.includes("error") || metric === "loss" ? left - right : right - left;
}
