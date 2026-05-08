"""공통 selection-tracked training loop 검증."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from methods.adaptation.common.selection_training_loop import (
    SelectionTrackedEpochResult,
    run_selection_tracked_training_loop,
)


def test_selection_tracked_training_loop_records_history_and_restores_best() -> None:
    model = nn.Linear(1, 1, bias=False)
    logs: list[str] = []
    accuracy_by_epoch = {1: 0.2, 2: 0.9, 3: 0.4}

    def train_epoch(epoch: int) -> SelectionTrackedEpochResult:
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        return SelectionTrackedEpochResult(
            train_loss_total=float(epoch),
            train_loss_denominator=2,
            extra_train_metrics={"train_aux": float(epoch)},
        )

    def evaluate_selection() -> dict[str, float]:
        epoch = int(model.weight.item())
        accuracy = accuracy_by_epoch[epoch]
        return {"loss": round(1.0 - accuracy, 6), "accuracy_top_1": accuracy}

    history, best_report = run_selection_tracked_training_loop(
        model=model,
        epochs=3,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message="missing best checkpoint",
        log_epoch_summary=logs.append,
    )

    assert [record["epoch"] for record in history] == [1, 2, 3]
    assert history[1]["selection_accuracy_top_1"] == 0.9
    assert history[2]["train_aux"] == 3.0
    assert best_report == {"loss": 0.1, "accuracy_top_1": 0.9}
    assert model.weight.item() == pytest.approx(2.0)
    assert logs[0].startswith("[epoch=1] train_loss=0.5000")


def test_selection_tracked_training_loop_requires_at_least_one_best_checkpoint() -> (
    None
):
    model = nn.Linear(1, 1)

    with pytest.raises(RuntimeError, match="missing best checkpoint"):
        run_selection_tracked_training_loop(
            model=model,
            epochs=0,
            train_epoch=lambda _epoch: SelectionTrackedEpochResult(
                train_loss_total=0.0,
                train_loss_denominator=1,
            ),
            evaluate_selection=lambda: {"loss": 1.0, "accuracy_top_1": 0.0},
            best_checkpoint_error_message="missing best checkpoint",
        )
