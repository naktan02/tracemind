"""임베딩 평가와 메트릭 계산."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from scripts.classification_report import (
    build_confusion_matrix,
    safe_divide,
    summarize_per_category,
)
from scripts.experiments.prototype_strategy.models import (
    EvaluationMetrics,
    PrototypeIndex,
    ScoredPrediction,
)
from scripts.experiments.prototype_strategy.scoring import (
    MaxCosinePrototypeIndexScorer,
)


def embed_rows(rows: Sequence[dict[str, Any]], adapter: Any) -> np.ndarray:
    """row의 text를 임베딩한다."""
    texts = [str(row["text"]) for row in rows]
    return np.asarray(adapter.embed_texts(texts), dtype=np.float64)


def group_embeddings_by_label(
    *,
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """row와 임베딩을 라벨별로 묶는다."""
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, embedding in zip(rows, embeddings, strict=True):
        buckets[str(row["mapped_label_4"])].append(embedding)
    return {
        label: np.asarray(label_embeddings, dtype=np.float64)
        for label, label_embeddings in sorted(buckets.items())
    }


def evaluate_embeddings(
    *,
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    prototype_index: PrototypeIndex,
    confidence_threshold: float,
    margin_threshold: float,
    scorer: MaxCosinePrototypeIndexScorer,
) -> EvaluationMetrics:
    """prototype 전략으로 평가셋 메트릭을 계산한다."""
    scored_predictions = score_embeddings(
        rows=rows,
        embeddings=embeddings,
        prototype_index=prototype_index,
        scorer=scorer,
    )
    return evaluate_scored_predictions(
        scored_predictions=scored_predictions,
        categories=sorted(prototype_index.categories.keys()),
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )


def score_embeddings(
    *,
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    prototype_index: PrototypeIndex,
    scorer: MaxCosinePrototypeIndexScorer,
) -> tuple[ScoredPrediction, ...]:
    """row/embedding으로부터 threshold 재평가 가능한 score 목록을 만든다."""
    categories = sorted(prototype_index.categories.keys())
    if not categories:
        raise ValueError("Prototype index must contain at least one category.")

    scored_predictions: list[ScoredPrediction] = []

    for row, embedding in zip(rows, embeddings, strict=True):
        scores = scorer.score(embedding, prototype_index)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top1_label, top1_score = ordered[0]
        top2_score = ordered[1][1] if len(ordered) > 1 else -1.0
        true_label = str(row["mapped_label_4"])
        margin = top1_score - top2_score

        scored_predictions.append(
            ScoredPrediction(
                actual_label=true_label,
                predicted_label=top1_label,
                true_label_score=float(scores[true_label]),
                top1_score=float(top1_score),
                top2_score=float(top2_score),
                margin_top1_top2=float(margin),
                is_correct=(true_label == top1_label),
            )
        )

    return tuple(scored_predictions)


def evaluate_scored_predictions(
    *,
    scored_predictions: Sequence[ScoredPrediction],
    categories: Sequence[str],
    confidence_threshold: float,
    margin_threshold: float,
) -> EvaluationMetrics:
    """미리 계산한 row별 score로 threshold를 재평가한다."""
    accepted_predictions = [
        prediction
        for prediction in scored_predictions
        if (
            prediction.top1_score >= confidence_threshold
            and prediction.margin_top1_top2 >= margin_threshold
        )
    ]
    return _build_evaluation_metrics(
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
    """전역 confidence threshold 하나로 accepted set을 평가한다."""
    accepted_predictions = [
        prediction
        for prediction in scored_predictions
        if prediction.top1_score >= confidence_threshold
    ]
    return _build_evaluation_metrics(
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
    """predicted label별 confidence threshold로 accepted set을 평가한다."""
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
    return _build_evaluation_metrics(
        scored_predictions=scored_predictions,
        categories=categories,
        accepted_predictions=accepted_predictions,
    )


def _build_evaluation_metrics(
    *,
    scored_predictions: Sequence[ScoredPrediction],
    categories: Sequence[str],
    accepted_predictions: Sequence[ScoredPrediction],
) -> EvaluationMetrics:
    """accepted prediction 집합이 주어졌을 때 공통 메트릭을 계산한다."""
    actual_labels = [prediction.actual_label for prediction in scored_predictions]
    predicted_labels = [
        prediction.predicted_label for prediction in scored_predictions
    ]
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


def build_per_category_metrics(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    true_label_scores: list[float],
    top1_scores: list[float],
    margins: list[float],
) -> dict[str, dict[str, float | int]]:
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
