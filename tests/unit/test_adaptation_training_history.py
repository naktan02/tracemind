"""공통 adaptation training history helper 검증."""

from __future__ import annotations

from methods.adaptation.common.training_history import (
    build_selection_epoch_record,
    format_selection_epoch_summary,
)


def test_build_selection_epoch_record_adds_common_selection_metrics() -> None:
    record = build_selection_epoch_record(
        epoch=2,
        train_loss_total=3.0,
        train_loss_denominator=2,
        selection_report={
            "loss": 0.25,
            "accuracy_top_1": 0.75,
        },
        extra_train_metrics={
            "train_unsup_loss": 0.4,
            "train_util_ratio": 0.5,
        },
    )

    assert record == {
        "epoch": 2,
        "train_loss": 1.5,
        "train_unsup_loss": 0.4,
        "train_util_ratio": 0.5,
        "selection_loss": 0.25,
        "selection_accuracy_top_1": 0.75,
    }


def test_format_selection_epoch_summary_keeps_training_metric_order() -> None:
    summary = format_selection_epoch_summary(
        {
            "epoch": 1,
            "train_loss": 0.123456,
            "train_sup_loss": 0.2,
            "selection_loss": 0.3,
            "selection_accuracy_top_1": 0.4,
        }
    )

    assert summary == (
        "train_loss=0.1235 train_sup_loss=0.2000 "
        "selection_loss=0.3000 selection_accuracy=0.4000"
    )
