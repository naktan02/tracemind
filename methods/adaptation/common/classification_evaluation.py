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
    aggregate_metrics = _build_paper_comparison_metrics(
        categories=categories,
        per_category=per_category,
        top_1_values=top_1_values,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
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
        **aggregate_metrics,
    }


def _build_paper_comparison_metrics(
    *,
    categories: list[str],
    per_category: dict[str, dict[str, float | int]],
    top_1_values: list[float],
    actual_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, float | str | None]:
    """논문 비교에서 직접 쓰는 aggregate 분류/calibration metric."""

    supports = {
        category: int(per_category[category]["support"]) for category in categories
    }
    total_support = sum(supports.values())
    macro_precision = _mean_metric(per_category, categories, "precision")
    macro_recall = _mean_metric(per_category, categories, "recall")
    macro_f1 = _mean_metric(per_category, categories, "f1")
    worst_f1_category = _worst_category(per_category, categories, "f1")
    calibration = _build_calibration_metrics(
        top_1_values=top_1_values,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )
    return {
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_precision": round(
            _weighted_metric(per_category, supports, total_support, "precision"),
            6,
        ),
        "weighted_recall": round(
            _weighted_metric(per_category, supports, total_support, "recall"),
            6,
        ),
        "weighted_f1": round(
            _weighted_metric(per_category, supports, total_support, "f1"),
            6,
        ),
        "balanced_accuracy": round(macro_recall, 6),
        "worst_category_f1": None
        if worst_f1_category is None
        else str(worst_f1_category),
        "worst_category_f1_value": None
        if worst_f1_category is None
        else float(per_category[worst_f1_category]["f1"]),
        "worst_category_recall": _worst_metric_value(
            per_category,
            categories,
            "recall",
        ),
        "worst_category_precision": _worst_metric_value(
            per_category,
            categories,
            "precision",
        ),
        **calibration,
    }


def _mean_metric(
    per_category: dict[str, dict[str, float | int]],
    categories: list[str],
    metric_key: str,
) -> float:
    if not categories:
        return 0.0
    return safe_divide(
        sum(float(per_category[category][metric_key]) for category in categories),
        len(categories),
    )


def _weighted_metric(
    per_category: dict[str, dict[str, float | int]],
    supports: dict[str, int],
    total_support: int,
    metric_key: str,
) -> float:
    return safe_divide(
        sum(
            float(per_category[category][metric_key]) * supports[category]
            for category in supports
        ),
        total_support,
    )


def _worst_category(
    per_category: dict[str, dict[str, float | int]],
    categories: list[str],
    metric_key: str,
) -> str | None:
    if not categories:
        return None
    return min(
        categories, key=lambda category: float(per_category[category][metric_key])
    )


def _worst_metric_value(
    per_category: dict[str, dict[str, float | int]],
    categories: list[str],
    metric_key: str,
) -> float | None:
    category = _worst_category(
        per_category=per_category,
        categories=categories,
        metric_key=metric_key,
    )
    if category is None:
        return None
    return float(per_category[category][metric_key])


def _build_calibration_metrics(
    *,
    top_1_values: list[float],
    actual_labels: list[str],
    predicted_labels: list[str],
    bin_count: int = 10,
) -> dict[str, float]:
    if not top_1_values:
        return {
            "expected_calibration_error": 0.0,
            "max_calibration_error": 0.0,
            "overconfidence_gap": 0.0,
            "mean_correct_top_1_probability": 0.0,
            "mean_incorrect_top_1_probability": 0.0,
        }

    correctness = [
        1.0 if actual == predicted else 0.0
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
    ]
    total = len(top_1_values)
    expected_calibration_error = 0.0
    max_calibration_error = 0.0
    for bin_index in range(bin_count):
        lower_bound = bin_index / bin_count
        upper_bound = (bin_index + 1) / bin_count
        selected = [
            index
            for index, confidence in enumerate(top_1_values)
            if (
                lower_bound <= confidence < upper_bound
                or (bin_index == bin_count - 1 and confidence == upper_bound)
            )
        ]
        if not selected:
            continue
        bin_accuracy = safe_divide(
            sum(correctness[index] for index in selected), len(selected)
        )
        bin_confidence = safe_divide(
            sum(top_1_values[index] for index in selected),
            len(selected),
        )
        calibration_gap = abs(bin_accuracy - bin_confidence)
        expected_calibration_error += (
            safe_divide(len(selected), total) * calibration_gap
        )
        max_calibration_error = max(max_calibration_error, calibration_gap)

    correct_confidences = [
        confidence
        for confidence, correct in zip(top_1_values, correctness, strict=True)
        if correct
    ]
    incorrect_confidences = [
        confidence
        for confidence, correct in zip(top_1_values, correctness, strict=True)
        if not correct
    ]
    accuracy = safe_divide(sum(correctness), total)
    mean_confidence = safe_divide(sum(top_1_values), total)
    return {
        "expected_calibration_error": round(expected_calibration_error, 6),
        "max_calibration_error": round(max_calibration_error, 6),
        "overconfidence_gap": round(mean_confidence - accuracy, 6),
        "mean_correct_top_1_probability": round(
            safe_divide(sum(correct_confidences), len(correct_confidences)),
            6,
        ),
        "mean_incorrect_top_1_probability": round(
            safe_divide(sum(incorrect_confidences), len(incorrect_confidences)),
            6,
        ),
    }
