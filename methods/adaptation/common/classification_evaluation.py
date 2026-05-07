"""분류 평가 report helper."""

from __future__ import annotations

from typing import Any

from shared.src.domain.services.classification_report import (
    build_confusion_matrix,
    safe_divide,
    summarize_per_category,
)


def build_classification_evaluation_report(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    true_probs: list[float],
    top_1_values: list[float],
    margins: list[float],
    total_loss: float,
    total_rows: int,
) -> dict[str, Any]:
    """분류 평가 report의 canonical metric shape를 만든다."""

    correct = sum(
        1
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    confusion_matrix = build_confusion_matrix(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )
    per_category = summarize_per_category(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        primary_values=true_probs,
        top_1_values=top_1_values,
        margins=margins,
        primary_metric_key="mean_true_label_probability",
        top_1_metric_key="mean_top_1_probability",
    )
    return {
        "rows_total": total_rows,
        "loss": round(safe_divide(total_loss, total_rows), 6),
        "accuracy_top_1": round(safe_divide(correct, total_rows), 6),
        "correct_top_1": correct,
        "mean_true_label_probability": round(
            safe_divide(sum(true_probs), len(true_probs)),
            6,
        ),
        "mean_top_1_probability": round(
            safe_divide(sum(top_1_values), len(top_1_values)),
            6,
        ),
        "mean_margin_top1_top2": round(
            safe_divide(sum(margins), len(margins)),
            6,
        ),
        "confusion_matrix": confusion_matrix,
        "per_category": per_category,
    }
