"""분류 학습 history record helper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.src.domain.services.classification_report import safe_divide


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
    return " ".join(metric_fields)
