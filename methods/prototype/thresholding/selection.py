"""Threshold policy 후보 선택 규칙."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from methods.prototype.thresholding.models import (
    ThresholdArtifact,
    ThresholdPolicyEvaluation,
)


@dataclass(slots=True)
class ThresholdPolicySelectionPolicy:
    """Validation 결과로 가장 적합한 threshold policy 후보를 고른다."""

    minimum_accepted_ratio: float = 0.5

    def select(
        self,
        evaluations: Sequence[ThresholdPolicyEvaluation],
    ) -> ThresholdPolicyEvaluation:
        if not evaluations:
            raise ValueError("At least one threshold policy evaluation is required.")
        eligible = tuple(
            evaluation
            for evaluation in evaluations
            if (
                evaluation.validation_metrics.accepted_ratio
                >= self.minimum_accepted_ratio
            )
        )
        candidates = eligible or tuple(evaluations)
        return max(candidates, key=self._selection_key)

    def sort(
        self,
        evaluations: Sequence[ThresholdPolicyEvaluation],
    ) -> tuple[ThresholdPolicyEvaluation, ...]:
        return tuple(
            sorted(
                evaluations,
                key=self._selection_key,
                reverse=True,
            )
        )

    @staticmethod
    def _selection_key(
        evaluation: ThresholdPolicyEvaluation,
    ) -> tuple[float | str, ...]:
        metrics = evaluation.validation_metrics
        return (
            metrics.accepted_accuracy,
            metrics.accepted_correct_ratio,
            metrics.accepted_ratio,
            _confidence_threshold_or_floor(evaluation.threshold_artifact),
            evaluation.policy_name,
            evaluation.candidate_name,
        )


def _confidence_threshold_or_floor(artifact: ThresholdArtifact) -> float:
    value = artifact.parameters.get("confidence_threshold")
    if isinstance(value, (int, float)):
        return float(value)
    classwise = artifact.parameters.get("confidence_threshold_by_label")
    if isinstance(classwise, dict) and classwise:
        numeric_values = [
            float(item) for item in classwise.values() if isinstance(item, (int, float))
        ]
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
    return -1.0
