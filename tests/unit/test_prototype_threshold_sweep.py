"""Prototype threshold sweep unit tests."""

from __future__ import annotations

from scripts.experiments.prototype_strategy.evaluation import evaluate_scored_predictions
from scripts.experiments.prototype_strategy.models import ScoredPrediction
from scripts.experiments.prototype_strategy.sweep import (
    ThresholdSweepPoint,
    ThresholdSweepSelectionPolicy,
)


def test_evaluate_scored_predictions_computes_accepted_metrics() -> None:
    predictions = (
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="anxiety",
            true_label_score=0.91,
            top1_score=0.91,
            top2_score=0.20,
            margin_top1_top2=0.71,
            is_correct=True,
        ),
        ScoredPrediction(
            actual_label="anxiety",
            predicted_label="depression",
            true_label_score=0.30,
            top1_score=0.82,
            top2_score=0.78,
            margin_top1_top2=0.04,
            is_correct=False,
        ),
        ScoredPrediction(
            actual_label="normal",
            predicted_label="normal",
            true_label_score=0.74,
            top1_score=0.74,
            top2_score=0.40,
            margin_top1_top2=0.34,
            is_correct=True,
        ),
    )

    metrics = evaluate_scored_predictions(
        scored_predictions=predictions,
        categories=("anxiety", "depression", "normal"),
        confidence_threshold=0.8,
        margin_threshold=0.1,
    )

    assert metrics.accepted_count == 1
    assert metrics.accepted_ratio == 1 / 3
    assert metrics.accepted_accuracy == 1.0
    assert metrics.accepted_correct_ratio == 1 / 3


def test_threshold_selection_prefers_more_correct_accepted_labels() -> None:
    policy = ThresholdSweepSelectionPolicy(minimum_accepted_ratio=0.05)

    def point(
        *,
        confidence: float,
        margin: float,
        accepted_correct_ratio: float,
        accepted_accuracy: float,
        accepted_ratio: float,
    ) -> ThresholdSweepPoint:
        metrics = evaluate_scored_predictions(
            scored_predictions=(),
            categories=(),
            confidence_threshold=confidence,
            margin_threshold=margin,
        )
        metrics.accepted_correct_ratio = accepted_correct_ratio
        metrics.accepted_accuracy = accepted_accuracy
        metrics.accepted_ratio = accepted_ratio
        return ThresholdSweepPoint(
            confidence_threshold=confidence,
            margin_threshold=margin,
            validation_metrics=metrics,
        )

    selected = policy.select(
        [
            point(
                confidence=0.8,
                margin=0.15,
                accepted_correct_ratio=0.01,
                accepted_accuracy=1.0,
                accepted_ratio=0.01,
            ),
            point(
                confidence=0.7,
                margin=0.03,
                accepted_correct_ratio=0.08,
                accepted_accuracy=0.92,
                accepted_ratio=0.09,
            ),
            point(
                confidence=0.65,
                margin=0.01,
                accepted_correct_ratio=0.06,
                accepted_accuracy=0.75,
                accepted_ratio=0.12,
            ),
        ]
    )

    assert selected.confidence_threshold == 0.7
    assert selected.margin_threshold == 0.03


def test_threshold_selection_prefers_precision_when_coverage_floor_is_met() -> None:
    policy = ThresholdSweepSelectionPolicy(minimum_accepted_ratio=0.5)

    def point(
        *,
        confidence: float,
        margin: float,
        accepted_correct_ratio: float,
        accepted_accuracy: float,
        accepted_ratio: float,
    ) -> ThresholdSweepPoint:
        metrics = evaluate_scored_predictions(
            scored_predictions=(),
            categories=(),
            confidence_threshold=confidence,
            margin_threshold=margin,
        )
        metrics.accepted_correct_ratio = accepted_correct_ratio
        metrics.accepted_accuracy = accepted_accuracy
        metrics.accepted_ratio = accepted_ratio
        return ThresholdSweepPoint(
            confidence_threshold=confidence,
            margin_threshold=margin,
            validation_metrics=metrics,
        )

    selected = policy.select(
        [
            point(
                confidence=0.6,
                margin=0.0,
                accepted_correct_ratio=0.6916,
                accepted_accuracy=0.7462,
                accepted_ratio=0.9268,
            ),
            point(
                confidence=0.6,
                margin=0.02,
                accepted_correct_ratio=0.4425,
                accepted_accuracy=0.8808,
                accepted_ratio=0.5023,
            ),
            point(
                confidence=0.6,
                margin=0.03,
                accepted_correct_ratio=0.3461,
                accepted_accuracy=0.9216,
                accepted_ratio=0.3755,
            ),
        ]
    )

    assert selected.confidence_threshold == 0.6
    assert selected.margin_threshold == 0.02
