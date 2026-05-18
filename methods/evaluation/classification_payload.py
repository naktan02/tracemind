"""Classification report를 track별 evaluation payload로 정규화한다."""

from __future__ import annotations

from collections.abc import Mapping


def build_classification_evaluation_payload(
    *,
    report: Mapping[str, object],
    row_count: int,
    accepted_ratio: float,
    loss_kind: str,
    score_distribution_kind: str,
    selection_confidence_kind: str | None,
    mean_selection_confidence: float,
    mean_selection_margin: float,
) -> dict[str, object]:
    """classification report를 dashboard/report용 canonical shape로 만든다."""

    return {
        "row_count": row_count,
        "top1_accuracy": float(report["accuracy_top_1"]),
        "accepted_ratio": accepted_ratio,
        "loss": float(report["loss"]),
        "loss_kind": loss_kind,
        "accuracy_top_1": float(report["accuracy_top_1"]),
        "correct_top_1": int(report["correct_top_1"]),
        "macro_f1": float(report["macro_f1"]),
        "macro_precision": float(report["macro_precision"]),
        "macro_recall": float(report["macro_recall"]),
        "weighted_f1": float(report["weighted_f1"]),
        "balanced_accuracy": float(report["balanced_accuracy"]),
        "worst_category_f1": _optional_str(report["worst_category_f1"]),
        "worst_category_f1_value": _optional_float(report["worst_category_f1_value"]),
        "worst_category_recall": _optional_float(report["worst_category_recall"]),
        "worst_category_precision": _optional_float(report["worst_category_precision"]),
        "expected_calibration_error": float(report["expected_calibration_error"]),
        "max_calibration_error": float(report["max_calibration_error"]),
        "overconfidence_gap": float(report["overconfidence_gap"]),
        "mean_true_label_probability": float(report["mean_true_label_probability"]),
        "mean_top_1_probability": float(report["mean_top_1_probability"]),
        "mean_margin_top1_top2": float(report["mean_margin_top1_top2"]),
        "mean_correct_top_1_probability": float(
            report["mean_correct_top_1_probability"]
        ),
        "mean_incorrect_top_1_probability": float(
            report["mean_incorrect_top_1_probability"]
        ),
        "score_distribution_kind": score_distribution_kind,
        "selection_confidence_kind": selection_confidence_kind,
        "mean_selection_confidence": mean_selection_confidence,
        "mean_selection_margin": mean_selection_margin,
        "per_label": typed_per_label(report["per_category"]),
        "confusion_matrix": typed_confusion_matrix(report["confusion_matrix"]),
        "classification_report": dict(report),
    }


def typed_per_label(value: object) -> dict[str, dict[str, int | float]]:
    """classification report per-category payload를 typed mapping으로 고정한다."""

    if not isinstance(value, dict):
        raise TypeError("per_category must be a dict.")
    return {
        str(label): dict(metrics)
        for label, metrics in value.items()
        if isinstance(metrics, dict)
    }


def typed_confusion_matrix(value: object) -> dict[str, dict[str, int]]:
    """classification report confusion matrix를 typed mapping으로 고정한다."""

    if not isinstance(value, dict):
        raise TypeError("confusion_matrix must be a dict.")
    return {
        str(actual): {str(predicted): int(count) for predicted, count in row.items()}
        for actual, row in value.items()
        if isinstance(row, dict)
    }


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
