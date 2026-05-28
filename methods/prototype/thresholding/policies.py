"""정적 threshold policy 구현체."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from methods.prototype.thresholding.evaluation import (
    evaluate_classwise_confidence_threshold,
    evaluate_global_confidence_threshold,
)
from methods.prototype.thresholding.models import (
    EvaluationMetrics,
    PaperReference,
    ScoredPrediction,
    ThresholdArtifact,
    ThresholdPolicyEvaluation,
)


class StaticThresholdPolicy(Protocol):
    """Validation score로 정적 threshold를 정하는 정책."""

    policy_name: str
    source_paper: PaperReference

    def build_evaluations(
        self,
        *,
        validation_predictions: Sequence[ScoredPrediction],
        test_predictions: Sequence[ScoredPrediction],
        categories: Sequence[str],
    ) -> tuple[ThresholdPolicyEvaluation, ...]:
        """Validation/test prediction으로 정책 후보들을 평가한다."""


@dataclass(slots=True)
class FixMatchFixedConfidencePolicy:
    """FixMatch식 전역 confidence threshold 후보군."""

    thresholds: tuple[float, ...] = (0.8, 0.9, 0.95)
    policy_name: str = "fixmatch_fixed_confidence"
    source_paper: PaperReference = field(
        default_factory=lambda: PaperReference(
            title=(
                "FixMatch: Simplifying Semi-Supervised Learning with "
                "Consistency and Confidence"
            ),
            url="https://openreview.net/forum?id=bleIdqV_-JY",
            venue="NeurIPS",
            year=2020,
        )
    )

    def build_evaluations(
        self,
        *,
        validation_predictions: Sequence[ScoredPrediction],
        test_predictions: Sequence[ScoredPrediction],
        categories: Sequence[str],
    ) -> tuple[ThresholdPolicyEvaluation, ...]:
        evaluations: list[ThresholdPolicyEvaluation] = []
        for threshold in self.thresholds:
            artifact = ThresholdArtifact(
                threshold_kind="global_confidence",
                parameters={"confidence_threshold": float(threshold)},
            )
            validation_metrics = evaluate_global_confidence_threshold(
                scored_predictions=validation_predictions,
                categories=categories,
                confidence_threshold=float(threshold),
            )
            test_metrics = evaluate_global_confidence_threshold(
                scored_predictions=test_predictions,
                categories=categories,
                confidence_threshold=float(threshold),
            )
            evaluations.append(
                ThresholdPolicyEvaluation(
                    policy_name=self.policy_name,
                    candidate_name=f"confidence={float(threshold):.3f}",
                    source_paper=self.source_paper,
                    selection_params={"confidence_threshold": float(threshold)},
                    threshold_artifact=artifact,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                )
            )
        return tuple(evaluations)


@dataclass(slots=True)
class ValidationTargetErrorConfidencePolicy:
    """Validation target error를 만족하는 최대 coverage threshold."""

    target_errors: tuple[float, ...] = (0.05, 0.1, 0.15)
    minimum_accepted_count: int = 1
    policy_name: str = "validation_target_error_confidence"
    source_paper: PaperReference = field(
        default_factory=lambda: PaperReference(
            title=(
                "Rethinking Confidence Scores and Thresholds in "
                "Pseudolabeling-based Semi-supervised Learning"
            ),
            url="https://proceedings.mlr.press/v267/vishwakarma25a.html",
            venue="ICML",
            year=2025,
        )
    )

    def build_evaluations(
        self,
        *,
        validation_predictions: Sequence[ScoredPrediction],
        test_predictions: Sequence[ScoredPrediction],
        categories: Sequence[str],
    ) -> tuple[ThresholdPolicyEvaluation, ...]:
        threshold_metrics = self._evaluate_threshold_candidates(
            predictions=validation_predictions,
            categories=categories,
        )
        evaluations: list[ThresholdPolicyEvaluation] = []
        for target_error in self.target_errors:
            selected_threshold, validation_metrics = self._select_threshold_for_error(
                threshold_metrics=threshold_metrics,
                target_error=float(target_error),
            )
            artifact = ThresholdArtifact(
                threshold_kind="global_confidence",
                parameters={
                    "confidence_threshold": selected_threshold,
                    "fit_objective": "target_error",
                    "target_error": float(target_error),
                },
            )
            test_metrics = evaluate_global_confidence_threshold(
                scored_predictions=test_predictions,
                categories=categories,
                confidence_threshold=selected_threshold,
            )
            evaluations.append(
                ThresholdPolicyEvaluation(
                    policy_name=self.policy_name,
                    candidate_name=f"target_error={float(target_error):.3f}",
                    source_paper=self.source_paper,
                    selection_params={"target_error": float(target_error)},
                    threshold_artifact=artifact,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                )
            )
        return tuple(evaluations)

    def _evaluate_threshold_candidates(
        self,
        *,
        predictions: Sequence[ScoredPrediction],
        categories: Sequence[str],
    ) -> tuple[tuple[float, EvaluationMetrics], ...]:
        thresholds = sorted(
            {float(prediction.top1_score) for prediction in predictions},
            reverse=True,
        )
        if not thresholds:
            raise ValueError("At least one validation prediction is required.")
        return tuple(
            (
                threshold,
                evaluate_global_confidence_threshold(
                    scored_predictions=predictions,
                    categories=categories,
                    confidence_threshold=threshold,
                ),
            )
            for threshold in thresholds
        )

    def _select_threshold_for_error(
        self,
        *,
        threshold_metrics: Sequence[tuple[float, EvaluationMetrics]],
        target_error: float,
    ) -> tuple[float, EvaluationMetrics]:
        feasible = [
            (threshold, metrics)
            for threshold, metrics in threshold_metrics
            if (
                metrics.accepted_count >= self.minimum_accepted_count
                and _accepted_error(metrics) <= target_error
            )
        ]
        if feasible:
            selected_threshold, selected_metrics = max(
                feasible,
                key=lambda item: (
                    item[1].accepted_ratio,
                    item[1].accepted_accuracy,
                    item[0],
                ),
            )
            return float(selected_threshold), selected_metrics

        non_empty = [
            (threshold, metrics)
            for threshold, metrics in threshold_metrics
            if metrics.accepted_count >= self.minimum_accepted_count
        ]
        if not non_empty:
            raise ValueError("At least one non-empty threshold candidate is required.")
        selected_threshold, selected_metrics = min(
            non_empty,
            key=lambda item: (
                _accepted_error(item[1]),
                -item[1].accepted_ratio,
                -item[1].accepted_accuracy,
                -item[0],
            ),
        )
        return float(selected_threshold), selected_metrics


@dataclass(slots=True)
class ClasswiseStaticConfidencePolicy:
    """predicted label별 validation target error threshold를 정적으로 fit한다."""

    target_errors: tuple[float, ...] = (0.1,)
    minimum_accepted_count: int = 1
    default_confidence_threshold: float = 1.0
    policy_name: str = "classwise_static_confidence"
    source_paper: PaperReference = field(
        default_factory=lambda: PaperReference(
            title=(
                "Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding"
            ),
            url="https://proceedings.mlr.press/v162/guo22e.html",
            venue="ICML",
            year=2022,
        )
    )

    def build_evaluations(
        self,
        *,
        validation_predictions: Sequence[ScoredPrediction],
        test_predictions: Sequence[ScoredPrediction],
        categories: Sequence[str],
    ) -> tuple[ThresholdPolicyEvaluation, ...]:
        evaluations: list[ThresholdPolicyEvaluation] = []
        predictions_by_label = _group_predictions_by_predicted_label(
            validation_predictions
        )
        for target_error in self.target_errors:
            threshold_by_label: dict[str, float] = {}
            for label in categories:
                label_predictions = predictions_by_label.get(label, ())
                if not label_predictions:
                    threshold_by_label[label] = self.default_confidence_threshold
                    continue
                threshold_metrics = self._evaluate_threshold_candidates(
                    predictions=label_predictions,
                    categories=categories,
                )
                threshold_by_label[label], _ = self._select_threshold_for_error(
                    threshold_metrics=threshold_metrics,
                    target_error=float(target_error),
                )

            artifact = ThresholdArtifact(
                threshold_kind="classwise_confidence",
                parameters={
                    "confidence_threshold_by_label": dict(
                        sorted(threshold_by_label.items())
                    ),
                    "fit_objective": "target_error",
                    "target_error": float(target_error),
                    "default_confidence_threshold": float(
                        self.default_confidence_threshold
                    ),
                },
            )
            validation_metrics = evaluate_classwise_confidence_threshold(
                scored_predictions=validation_predictions,
                categories=categories,
                confidence_threshold_by_label=threshold_by_label,
                default_confidence_threshold=self.default_confidence_threshold,
            )
            test_metrics = evaluate_classwise_confidence_threshold(
                scored_predictions=test_predictions,
                categories=categories,
                confidence_threshold_by_label=threshold_by_label,
                default_confidence_threshold=self.default_confidence_threshold,
            )
            evaluations.append(
                ThresholdPolicyEvaluation(
                    policy_name=self.policy_name,
                    candidate_name=f"target_error={float(target_error):.3f}",
                    source_paper=self.source_paper,
                    selection_params={"target_error": float(target_error)},
                    threshold_artifact=artifact,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                )
            )
        return tuple(evaluations)

    def _evaluate_threshold_candidates(
        self,
        *,
        predictions: Sequence[ScoredPrediction],
        categories: Sequence[str],
    ) -> tuple[tuple[float, EvaluationMetrics], ...]:
        thresholds = sorted(
            {float(prediction.top1_score) for prediction in predictions},
            reverse=True,
        )
        if not thresholds:
            raise ValueError("At least one validation prediction is required.")
        return tuple(
            (
                threshold,
                evaluate_global_confidence_threshold(
                    scored_predictions=predictions,
                    categories=categories,
                    confidence_threshold=threshold,
                ),
            )
            for threshold in thresholds
        )

    def _select_threshold_for_error(
        self,
        *,
        threshold_metrics: Sequence[tuple[float, EvaluationMetrics]],
        target_error: float,
    ) -> tuple[float, EvaluationMetrics]:
        feasible = [
            (threshold, metrics)
            for threshold, metrics in threshold_metrics
            if (
                metrics.accepted_count >= self.minimum_accepted_count
                and _accepted_error(metrics) <= target_error
            )
        ]
        if feasible:
            selected_threshold, selected_metrics = max(
                feasible,
                key=lambda item: (
                    item[1].accepted_ratio,
                    item[1].accepted_accuracy,
                    item[0],
                ),
            )
            return float(selected_threshold), selected_metrics

        non_empty = [
            (threshold, metrics)
            for threshold, metrics in threshold_metrics
            if metrics.accepted_count >= self.minimum_accepted_count
        ]
        if not non_empty:
            raise ValueError("At least one non-empty threshold candidate is required.")
        selected_threshold, selected_metrics = min(
            non_empty,
            key=lambda item: (
                _accepted_error(item[1]),
                -item[1].accepted_ratio,
                -item[1].accepted_accuracy,
                -item[0],
            ),
        )
        return float(selected_threshold), selected_metrics


def _accepted_error(metrics: EvaluationMetrics) -> float:
    if metrics.accepted_count <= 0:
        return 1.0
    return 1.0 - metrics.accepted_accuracy


def _group_predictions_by_predicted_label(
    predictions: Sequence[ScoredPrediction],
) -> dict[str, tuple[ScoredPrediction, ...]]:
    buckets: dict[str, list[ScoredPrediction]] = {}
    for prediction in predictions:
        buckets.setdefault(prediction.predicted_label, []).append(prediction)
    return {
        label: tuple(label_predictions)
        for label, label_predictions in sorted(buckets.items())
    }
