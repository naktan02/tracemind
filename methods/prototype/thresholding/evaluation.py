"""Prototype scored prediction threshold evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from methods.prototype.thresholding.models import EvaluationMetrics, ScoredPrediction
from shared.src.domain.services.classification_report import (
    build_confusion_matrix,
    safe_divide,
    summarize_per_category,
)


def evaluate_scored_predictions(
    *,
    scored_predictions: Sequence[ScoredPrediction],
    categories: Sequence[str],
    confidence_threshold: float,
    margin_threshold: float,
) -> EvaluationMetrics:
    """Evaluate an accepted set from precomputed row scores."""

    accepted_predictions = [
        prediction
        for prediction in scored_predictions
        if (
            prediction.top1_score >= confidence_threshold
            and prediction.margin_top1_top2 >= margin_threshold
        )
    ]
    return build_evaluation_metrics(
        scored_predictions=scored_predictions,
        categories=categories,
        accepted_predictions=accepted_predictions,
    )


def evaluate_global_confidence_threshold(
    *,
    scored_predictions: Sequence[ScoredPrediction],
    categories: Sequence[str],
    confidence_threshold: float,
) -> EvaluationMetrics:
    """Evaluate an accepted set with one global confidence threshold."""

    accepted_predictions = [
        prediction
        for prediction in scored_predictions
        if prediction.top1_score >= confidence_threshold
    ]
    return build_evaluation_metrics(
        scored_predictions=scored_predictions,
        categories=categories,
        accepted_predictions=accepted_predictions,
    )


def evaluate_classwise_confidence_threshold(
    *,
    scored_predictions: Sequence[ScoredPrediction],
    categories: Sequence[str],
    confidence_threshold_by_label: dict[str, float],
    default_confidence_threshold: float | None = None,
) -> EvaluationMetrics:
    """Evaluate an accepted set with predicted-label thresholds."""

    accepted_predictions = [
        prediction
        for prediction in scored_predictions
        if prediction.top1_score
        >= _resolve_classwise_threshold(
            predicted_label=prediction.predicted_label,
            confidence_threshold_by_label=confidence_threshold_by_label,
            default_confidence_threshold=default_confidence_threshold,
        )
    ]
    return build_evaluation_metrics(
        scored_predictions=scored_predictions,
        categories=categories,
        accepted_predictions=accepted_predictions,
    )


def build_evaluation_metrics(
    *,
    scored_predictions: Sequence[ScoredPrediction],
    categories: Sequence[str],
    accepted_predictions: Sequence[ScoredPrediction],
) -> EvaluationMetrics:
    """Build common metrics for a scored prediction set."""

    actual_labels = [prediction.actual_label for prediction in scored_predictions]
    predicted_labels = [prediction.predicted_label for prediction in scored_predictions]
    true_label_scores = [
        prediction.true_label_score for prediction in scored_predictions
    ]
    top1_scores = [prediction.top1_score for prediction in scored_predictions]
    margins = [prediction.margin_top1_top2 for prediction in scored_predictions]

    correct = sum(1 for prediction in scored_predictions if prediction.is_correct)
    accepted_correct = sum(
        1 for prediction in accepted_predictions if prediction.is_correct
    )
    accepted_count = len(accepted_predictions)

    return EvaluationMetrics(
        row_count=len(scored_predictions),
        top1_accuracy=safe_divide(correct, len(scored_predictions)),
        accepted_ratio=safe_divide(accepted_count, len(scored_predictions)),
        mean_true_label_score=safe_divide(
            sum(true_label_scores),
            len(true_label_scores),
        ),
        mean_top1_score=safe_divide(sum(top1_scores), len(top1_scores)),
        mean_margin_top1_top2=safe_divide(sum(margins), len(margins)),
        confusion_matrix=build_confusion_matrix(
            categories=list(categories),
            actual_labels=actual_labels,
            predicted_labels=predicted_labels,
        ),
        per_category=build_per_category_metrics(
            categories=list(categories),
            actual_labels=actual_labels,
            predicted_labels=predicted_labels,
            true_label_scores=true_label_scores,
            top1_scores=top1_scores,
            margins=margins,
        ),
        accepted_count=accepted_count,
        accepted_accuracy=safe_divide(accepted_correct, accepted_count),
        accepted_correct_ratio=safe_divide(accepted_correct, len(scored_predictions)),
        accepted_mean_top1_score=safe_divide(
            sum(prediction.top1_score for prediction in accepted_predictions),
            accepted_count,
        ),
        accepted_mean_margin_top1_top2=safe_divide(
            sum(prediction.margin_top1_top2 for prediction in accepted_predictions),
            accepted_count,
        ),
    )


def build_per_category_metrics(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    true_label_scores: list[float],
    top1_scores: list[float],
    margins: list[float],
) -> dict[str, dict[str, float | int]]:
    """Build category-level accuracy and score summaries."""

    return summarize_per_category(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        primary_values=true_label_scores,
        top_1_values=top1_scores,
        margins=margins,
        primary_metric_key="mean_true_label_score",
        top_1_metric_key="mean_top1_score",
        round_digits=None,
    )


def _resolve_classwise_threshold(
    *,
    predicted_label: str,
    confidence_threshold_by_label: dict[str, float],
    default_confidence_threshold: float | None,
) -> float:
    if predicted_label in confidence_threshold_by_label:
        return float(confidence_threshold_by_label[predicted_label])
    if default_confidence_threshold is not None:
        return float(default_confidence_threshold)
    raise ValueError(
        "Missing classwise confidence threshold for predicted label: "
        f"{predicted_label}."
    )
