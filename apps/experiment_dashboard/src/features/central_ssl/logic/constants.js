export const CENTRAL_SSL_TRACK = "central_peft_ssl";
export const CENTRAL_INITIAL_EVAL_TRACK = "central_peft_initial_eval";

export const CENTRAL_EPOCH_METRICS = [
  "selection_macro_f1",
  "selection_accuracy_top_1",
  "selection_expected_calibration_error",
  "selection_worst_category_f1_value",
  "selection_worst_category_f1",
  "selection_loss",
  "train_loss",
  "train_sup_loss",
  "train_unsup_loss",
  "train_util_ratio",
];

export const CENTRAL_INITIAL_METRIC_MAP = {
  selection_accuracy_top_1: "accuracy_top_1",
  selection_macro_f1: "macro_f1",
  selection_expected_calibration_error: "expected_calibration_error",
  selection_worst_category_f1_value: "worst_category_f1_value",
  selection_loss: "loss",
};

export const CENTRAL_OVERVIEW_METRICS = [
  "macro_f1",
  "accuracy_top_1",
  "loss",
  "expected_calibration_error",
  "weighted_f1",
  "balanced_accuracy",
  "worst_category_f1_value",
  "max_calibration_error",
  "rows_total",
];

export const DEFAULT_CENTRAL_OVERVIEW_METRICS = [
  "macro_f1",
  "accuracy_top_1",
  "loss",
  "expected_calibration_error",
];
