"""분류 학습 history record helper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.src.domain.services.classification_report import safe_divide

_OPTIONAL_SELECTION_METRICS = (
    ("macro_f1", "selection_macro_f1"),
    ("expected_calibration_error", "selection_expected_calibration_error"),
    ("worst_category_f1", "selection_worst_category_f1"),
    ("worst_category_f1_value", "selection_worst_category_f1_value"),
)


def build_selection_epoch_record(
    *,
    epoch: int,
    train_loss_total: float,
    train_loss_denominator: int,
    selection_report: Mapping[str, Any],
    extra_train_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """selection set 평가를 포함한 epoch history record를 만든다."""

    record: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": round(
            safe_divide(train_loss_total, train_loss_denominator),
            6,
        ),
    }
    if extra_train_metrics:
        record.update(extra_train_metrics)
    record.update(
        {
            "selection_loss": selection_report["loss"],
            "selection_accuracy_top_1": selection_report["accuracy_top_1"],
        }
    )
    for source_key, target_key in _OPTIONAL_SELECTION_METRICS:
        if source_key in selection_report:
            record[target_key] = selection_report[source_key]
    if "per_category" in selection_report:
        record["selection_per_category"] = selection_report["per_category"]
    return record


def format_selection_epoch_summary(epoch_record: Mapping[str, Any]) -> str:
    """epoch history record를 logging용 한 줄 summary로 변환한다."""

    metric_fields = [
        f"{name}={float(value):.4f}"
        for name, value in epoch_record.items()
        if name.startswith("train_")
    ]
    metric_fields.extend(
        [
            f"selection_loss={float(epoch_record['selection_loss']):.4f}",
            f"selection_accuracy={float(epoch_record['selection_accuracy_top_1']):.4f}",
        ]
    )
    if "selection_macro_f1" in epoch_record:
        metric_fields.append(
            f"selection_macro_f1={float(epoch_record['selection_macro_f1']):.4f}"
        )
    if "selection_expected_calibration_error" in epoch_record:
        metric_fields.append(
            "selection_ece="
            f"{float(epoch_record['selection_expected_calibration_error']):.4f}"
        )
    return " ".join(metric_fields)
