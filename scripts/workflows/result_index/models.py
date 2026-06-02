"""Experiment result index record models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentRunRecord:
    run_id: str
    track: str
    method_family: str
    method_name: str
    algorithm_name: str | None
    selection_slug: str | None
    labeled_dataset_name: str | None
    unlabeled_dataset_name: str | None
    validation_dataset_name: str | None
    test_dataset_name: str | None
    seed: int | None
    learning_rate: float | None
    classifier_learning_rate: float | None
    epochs: int | None
    max_train_steps: int | None
    train_batch_size: int | None
    eval_batch_size: int | None
    initial_checkpoint_name: str | None
    unlabeled_row_count: int | None
    total_row_exposure_count: int | None
    labeled_row_exposure_count: int | None
    unlabeled_row_exposure_count: int | None
    unique_total_row_count: int | None
    unique_labeled_row_count: int | None
    unique_unlabeled_row_count: int | None
    train_seconds: float | None
    training_example_count: int | None
    examples_per_second: float | None
    trainable_param_ratio: float | None
    peft_adapter_name: str | None
    peft_adapter_rank: int | None
    peft_adapter_alpha: int | None
    peft_adapter_dropout: float | None
    peft_adapter_bias: str | None
    peft_adapter_target_modules: str | None
    peft_adapter_parameters_json: str | None
    run_control_budget_name: str | None
    run_control_output_dir: str | None
    client_count: int | None
    round_budget: int | None
    completed_rounds: int | None
    shard_policy_name: str | None
    shard_alpha: float | None
    payload_adapter_kind: str | None
    update_family_name: str | None
    aggregation_backend_name: str | None
    fl_composition_mode: str | None
    fl_execution_role: str | None
    fl_descriptor_name: str | None
    update_delta_format: str | None
    local_regularizer_name: str | None
    local_regularizer_mu: float | None
    embedding_backend: str | None
    embedding_model_id: str | None
    embedding_device: str | None
    local_trainer_device: str | None
    created_at: str | None


@dataclass(frozen=True, slots=True)
class EvalMetricRecord:
    run_id: str
    eval_set: str
    rows_total: int | None
    loss: float | None
    accuracy_top_1: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    weighted_precision: float | None
    weighted_recall: float | None
    weighted_f1: float | None
    balanced_accuracy: float | None
    expected_calibration_error: float | None
    max_calibration_error: float | None
    overconfidence_gap: float | None
    worst_category_f1: str | None
    worst_category_f1_value: float | None
    worst_category_precision: float | None
    worst_category_recall: float | None
    mean_true_label_probability: float | None
    mean_top_1_probability: float | None
    mean_margin_top1_top2: float | None
    correct_top_1: int | None


@dataclass(frozen=True, slots=True)
class PerClassMetricRecord:
    run_id: str
    eval_set: str
    category: str
    support: int | None
    predicted: int | None
    correct: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    mean_true_label_probability: float | None
    mean_top_1_probability: float | None
    mean_margin_top1_top2: float | None


@dataclass(frozen=True, slots=True)
class ConfusionMatrixCellRecord:
    run_id: str
    eval_set: str
    actual_category: str
    predicted_category: str
    count: int


@dataclass(frozen=True, slots=True)
class EpochMetricRecord:
    run_id: str
    epoch: int
    step: int | None
    train_loss: float | None
    train_sup_loss: float | None
    train_unsup_loss: float | None
    train_util_ratio: float | None
    selection_loss: float | None
    selection_accuracy_top_1: float | None
    selection_macro_f1: float | None
    selection_expected_calibration_error: float | None
    selection_worst_category_f1: str | None
    selection_worst_category_f1_value: float | None


@dataclass(frozen=True, slots=True)
class EpochPerClassMetricRecord:
    run_id: str
    epoch: int
    category: str
    support: int | None
    predicted: int | None
    correct: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    mean_true_label_probability: float | None
    mean_top_1_probability: float | None
    mean_margin_top1_top2: float | None


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    run_id: str
    eval_set: str | None
    artifact_kind: str
    artifact_ref: str
    reducer: str | None
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class ResultIndexRecords:
    run: ExperimentRunRecord
    eval_metrics: tuple[EvalMetricRecord, ...]
    per_class_metrics: tuple[PerClassMetricRecord, ...]
    confusion_matrix_cells: tuple[ConfusionMatrixCellRecord, ...]
    epoch_metrics: tuple[EpochMetricRecord, ...]
    epoch_per_class_metrics: tuple[EpochPerClassMetricRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]
