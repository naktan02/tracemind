"""Federated simulation용 example build와 validation 계산."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from scripts.io.labeled_query_rows import LabeledQueryRow
from scripts.runtime_adapters.federated_agent_runtime import (
    build_federated_scoring_service,
    build_federated_training_examples,
    select_federated_pseudo_labels,
)
from shared.src.config.training_defaults import (
    build_default_training_objective_config,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

from .flow_models import ActiveSimulationState
from .io_utils import parse_created_at
from .models import (
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationRunRequest,
)


def build_training_examples(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    adapter_state: SharedAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: Any,
    objective_config: TrainingObjectiveConfig | None = None,
) -> tuple[Any, ...]:
    """simulation row를 agent runtime training example builder로 변환한다."""
    return build_federated_training_examples(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=prototype_pack,
        model_id=model_id,
        scoring_service=scoring_service,
        objective_config=objective_config,
        parse_created_at=parse_created_at,
    )


def evaluate_rows(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    adapter_state: SharedAdapterState,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scoring_service: Any,
    confidence_threshold: float,
    margin_threshold: float,
    objective_config: TrainingObjectiveConfig | None = None,
) -> SimulationEvaluation:
    """validation row에 대해 top1 accuracy와 pseudo-label acceptance 비율을 계산한다."""
    effective_objective_config = _build_validation_objective_config(
        objective_config=objective_config,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    examples = build_training_examples(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=prototype_pack,
        model_id=model_id,
        scoring_service=scoring_service,
        objective_config=effective_objective_config,
    )
    if not examples:
        return SimulationEvaluation(row_count=0, top1_accuracy=0.0, accepted_ratio=0.0)

    selection_result = select_federated_pseudo_labels(
        scored_events=tuple(example.evidence_scored_event for example in examples),
        training_task=_build_validation_task(
            model_id=model_id,
            model_revision=adapter_state.model_revision,
            training_scope=adapter_state.training_scope,
            objective_config=effective_objective_config,
        ),
    )
    true_labels = [str(row["mapped_label_4"]) for row in rows]
    predicted_labels = [candidate.label for candidate in selection_result.candidates]
    correctness = [
        predicted == true
        for true, predicted in zip(true_labels, predicted_labels, strict=True)
    ]
    macro_f1, per_label, confusion_matrix = _build_classification_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )

    return SimulationEvaluation(
        row_count=len(rows),
        top1_accuracy=sum(correctness) / len(rows),
        accepted_ratio=selection_result.accepted_ratio,
        macro_f1=macro_f1,
        expected_calibration_error=_compute_expected_calibration_error(
            confidences=[
                candidate.confidence for candidate in selection_result.candidates
            ],
            correctness=correctness,
        ),
        per_label=per_label,
        confusion_matrix=confusion_matrix,
    )


def build_validation_scoring_service(
    validation_config: FederatedValidationConfig,
    *,
    shared_state: SharedAdapterState | None = None,
) -> Any:
    """validation 설정으로 scoring service를 조립한다."""
    overrides: dict[str, str | int | float | bool] = {
        "scorer_backend_name": validation_config.scorer_backend_name,
    }
    if validation_config.score_policy_name is not None:
        overrides["score_policy_name"] = validation_config.score_policy_name
    if validation_config.score_top_k is not None:
        overrides["score_top_k"] = validation_config.score_top_k
    return build_federated_scoring_service(
        objective_config=build_default_training_objective_config(overrides=overrides),
        similarity_name=validation_config.similarity_name,
        shared_state=shared_state,
    )


def evaluate_simulation_validation(
    *,
    request: SimulationRunRequest,
    adapter: EmbeddingAdapter,
    active: ActiveSimulationState,
    rows: list[LabeledQueryRow],
    objective_config: TrainingObjectiveConfig | None,
) -> SimulationEvaluation:
    """FL simulation request 기준 validation row를 평가한다."""

    return evaluate_rows(
        rows=rows,
        adapter=adapter,
        adapter_state=active.adapter_state,
        prototype_pack=active.prototype_pack,
        model_id=request.model_id,
        scoring_service=build_validation_scoring_service(
            request.validation_config,
            shared_state=active.adapter_state,
        ),
        confidence_threshold=request.validation_config.confidence_threshold,
        margin_threshold=request.validation_config.margin_threshold,
        objective_config=objective_config,
    )


def _build_validation_objective_config(
    *,
    objective_config: TrainingObjectiveConfig | None,
    confidence_threshold: float,
    margin_threshold: float,
) -> TrainingObjectiveConfig:
    overrides: dict[str, str | int | float | bool] = (
        {} if objective_config is None else objective_config.to_mapping()
    )
    overrides["confidence_threshold"] = confidence_threshold
    overrides["margin_threshold"] = margin_threshold
    return build_default_training_objective_config(overrides=overrides)


def _build_validation_task(
    *,
    model_id: str,
    model_revision: str,
    training_scope: str,
    objective_config: TrainingObjectiveConfig,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="simulation_validation_task",
        round_id="simulation_validation_round",
        model_id=model_id,
        model_revision=model_revision,
        task_type="pseudo_label_self_training",
        training_scope=training_scope,
        local_epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        max_steps=1,
        objective_config=objective_config,
        selection_policy=TrainingSelectionPolicy(),
    )


def _build_classification_metrics(
    *,
    true_labels: list[str],
    predicted_labels: list[str],
) -> tuple[float, dict[str, dict[str, int | float]], dict[str, dict[str, int]]]:
    labels = sorted(set(true_labels) | set(predicted_labels))
    confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True):
        confusion[true_label][predicted_label] += 1

    confusion_matrix = {
        label: dict(sorted(counts.items()))
        for label, counts in sorted(confusion.items())
    }
    per_label: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = confusion[label].get(label, 0)
        false_positive = sum(
            confusion[other_label].get(label, 0)
            for other_label in labels
            if other_label != label
        )
        false_negative = sum(
            count
            for predicted_label, count in confusion[label].items()
            if predicted_label != label
        )
        support = true_positive + false_negative
        predicted_count = true_positive + false_positive
        precision = true_positive / predicted_count if predicted_count > 0 else 0.0
        recall = true_positive / support if support > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        per_label[label] = {
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return macro_f1, per_label, confusion_matrix


def _compute_expected_calibration_error(
    *,
    confidences: list[float],
    correctness: list[bool],
    bin_count: int = 10,
) -> float:
    if not confidences:
        return 0.0
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have the same length.")

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(bin_count)]
    for confidence, is_correct in zip(confidences, correctness, strict=True):
        clamped_confidence = min(1.0, max(0.0, confidence))
        bin_index = min(int(clamped_confidence * bin_count), bin_count - 1)
        bins[bin_index].append((clamped_confidence, is_correct))

    total = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bucket_confidence = sum(item[0] for item in bucket) / len(bucket)
        bucket_accuracy = sum(1 for _, item_correct in bucket if item_correct) / len(
            bucket
        )
        ece += (len(bucket) / total) * abs(bucket_accuracy - bucket_confidence)
    return ece
