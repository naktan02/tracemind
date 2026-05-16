"""Federated simulation용 example build와 validation 계산."""

from __future__ import annotations

import math
from typing import Any

from methods.evaluation.classification_report import (
    build_classification_evaluation_report,
)
from methods.federated_ssl.runtime_fallbacks import (
    build_runtime_fallback_training_objective_config,
)
from methods.prototype.evidence.helpers import softmax_distribution
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.io.rows import parse_created_at
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.scoring_runtime import (
    build_federated_scoring_service,
)
from scripts.runtime_adapters.federated_agent.selection_runtime import (
    select_federated_pseudo_labels,
)
from scripts.runtime_adapters.federated_agent.training_example_mapper import (
    build_federated_training_examples,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


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
    return _build_simulation_evaluation(
        row_count=len(rows),
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        candidates=list(selection_result.candidates),
        evidences=list(selection_result.evidences),
        accepted_ratio=selection_result.accepted_ratio,
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
        objective_config=build_runtime_fallback_training_objective_config(
            overrides=overrides
        ),
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
    return build_runtime_fallback_training_objective_config(overrides=overrides)


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


def _build_simulation_evaluation(
    *,
    row_count: int,
    true_labels: list[str],
    predicted_labels: list[str],
    candidates: list[Any],
    evidences: list[Any],
    accepted_ratio: float,
) -> SimulationEvaluation:
    categories = sorted(set(true_labels) | set(predicted_labels))
    distributions: list[dict[str, float]] = []
    distribution_kinds: list[str] = []
    for candidate, evidence in zip(candidates, evidences, strict=True):
        distribution, distribution_kind = _label_probability_distribution(
            candidate=candidate,
            evidence=evidence,
        )
        distributions.append(distribution)
        distribution_kinds.append(distribution_kind)

    true_probs = [
        distribution.get(true_label, 0.0)
        for true_label, distribution in zip(true_labels, distributions, strict=True)
    ]
    top_1_values = [
        distribution.get(predicted_label, 0.0)
        for predicted_label, distribution in zip(
            predicted_labels,
            distributions,
            strict=True,
        )
    ]
    margins = [
        _probability_margin(distribution=distribution, predicted_label=predicted_label)
        for predicted_label, distribution in zip(
            predicted_labels,
            distributions,
            strict=True,
        )
    ]
    total_loss = sum(-math.log(max(probability, 1e-12)) for probability in true_probs)
    report = build_classification_evaluation_report(
        categories=categories,
        actual_labels=true_labels,
        predicted_labels=predicted_labels,
        true_probs=true_probs,
        top_1_values=top_1_values,
        margins=margins,
        total_loss=total_loss,
        total_rows=row_count,
    )
    return _evaluation_from_report(
        report=report,
        row_count=row_count,
        accepted_ratio=accepted_ratio,
        candidates=candidates,
        distribution_kind=_summarize_distribution_kind(distribution_kinds),
    )


def _label_probability_distribution(
    *,
    candidate: Any,
    evidence: Any,
) -> tuple[dict[str, float], str]:
    label_distribution = getattr(evidence, "label_distribution", None)
    if label_distribution:
        return (
            _float_distribution(label_distribution),
            "evidence_label_distribution",
        )

    raw_scores = getattr(evidence, "raw_scores", None)
    if raw_scores:
        return (
            softmax_distribution(raw_scores, temperature=1.0),
            "softmax_raw_scores_temperature_1.0",
        )

    sparse_scores = {str(candidate.label): float(candidate.confidence)}
    if candidate.runner_up_label is not None:
        sparse_scores[str(candidate.runner_up_label)] = float(
            candidate.runner_up_score or 0.0
        )
    if len(sparse_scores) > 1:
        distribution = softmax_distribution(sparse_scores, temperature=1.0)
    else:
        distribution = {str(candidate.label): 1.0}
    return (
        distribution,
        "softmax_candidate_top2_sparse_fallback",
    )


def _float_distribution(
    distribution: dict[str, float],
) -> dict[str, float]:
    return {
        str(label): float(probability) for label, probability in distribution.items()
    }


def _probability_margin(
    *,
    distribution: dict[str, float],
    predicted_label: str,
) -> float:
    top_value = float(distribution.get(predicted_label, 0.0))
    runner_up = max(
        (
            float(value)
            for label, value in distribution.items()
            if label != predicted_label
        ),
        default=0.0,
    )
    return top_value - runner_up


def _evaluation_from_report(
    *,
    report: dict[str, object],
    row_count: int,
    accepted_ratio: float,
    candidates: list[Any],
    distribution_kind: str,
) -> SimulationEvaluation:
    return SimulationEvaluation(
        row_count=row_count,
        top1_accuracy=float(report["accuracy_top_1"]),
        accepted_ratio=accepted_ratio,
        loss=float(report["loss"]),
        loss_kind="negative_log_likelihood_from_score_distribution",
        accuracy_top_1=float(report["accuracy_top_1"]),
        correct_top_1=int(report["correct_top_1"]),
        macro_f1=float(report["macro_f1"]),
        macro_precision=float(report["macro_precision"]),
        macro_recall=float(report["macro_recall"]),
        weighted_f1=float(report["weighted_f1"]),
        balanced_accuracy=float(report["balanced_accuracy"]),
        worst_category_f1=_optional_str(report["worst_category_f1"]),
        worst_category_f1_value=_optional_float(report["worst_category_f1_value"]),
        worst_category_recall=_optional_float(report["worst_category_recall"]),
        worst_category_precision=_optional_float(report["worst_category_precision"]),
        expected_calibration_error=float(report["expected_calibration_error"]),
        max_calibration_error=float(report["max_calibration_error"]),
        overconfidence_gap=float(report["overconfidence_gap"]),
        mean_true_label_probability=float(report["mean_true_label_probability"]),
        mean_top_1_probability=float(report["mean_top_1_probability"]),
        mean_margin_top1_top2=float(report["mean_margin_top1_top2"]),
        mean_correct_top_1_probability=float(report["mean_correct_top_1_probability"]),
        mean_incorrect_top_1_probability=float(
            report["mean_incorrect_top_1_probability"]
        ),
        score_distribution_kind=distribution_kind,
        selection_confidence_kind=_summarize_selection_confidence_kind(candidates),
        mean_selection_confidence=_mean(
            [float(candidate.confidence) for candidate in candidates]
        ),
        mean_selection_margin=_mean(
            [float(candidate.margin) for candidate in candidates]
        ),
        per_label=_typed_per_label(report["per_category"]),
        confusion_matrix=_typed_confusion_matrix(report["confusion_matrix"]),
        classification_report=dict(report),
    )


def _summarize_distribution_kind(distribution_kinds: list[str]) -> str:
    unique_kinds = sorted(set(distribution_kinds))
    if not unique_kinds:
        return "not_computed"
    if len(unique_kinds) == 1:
        return unique_kinds[0]
    return "mixed:" + ",".join(unique_kinds)


def _summarize_selection_confidence_kind(candidates: list[Any]) -> str | None:
    unique_kinds = sorted(
        {
            str(candidate.confidence_kind)
            for candidate in candidates
            if candidate.confidence_kind is not None
        }
    )
    if not unique_kinds:
        return None
    if len(unique_kinds) == 1:
        return unique_kinds[0]
    return "mixed:" + ",".join(unique_kinds)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _typed_per_label(value: object) -> dict[str, dict[str, int | float]]:
    if not isinstance(value, dict):
        raise TypeError("per_category must be a dict.")
    return {
        str(label): dict(metrics)
        for label, metrics in value.items()
        if isinstance(metrics, dict)
    }


def _typed_confusion_matrix(value: object) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        raise TypeError("confusion_matrix must be a dict.")
    return {
        str(actual): {str(predicted): int(count) for predicted, count in row.items()}
        for actual, row in value.items()
        if isinstance(row, dict)
    }
