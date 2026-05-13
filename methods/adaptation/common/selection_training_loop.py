"""selection set 평가를 포함한 분류 학습 loop scaffold."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from methods.adaptation.common.checkpointing import BestModelCheckpoint
from methods.adaptation.common.training_history import (
    build_selection_epoch_record,
    format_selection_epoch_summary,
)


@dataclass(frozen=True, slots=True)
class SelectionTrackedEpochResult:
    """한 epoch 학습 결과 중 공통 history/checkpoint가 소비하는 부분."""

    train_loss_total: float
    train_loss_denominator: int
    extra_train_metrics: Mapping[str, float] | None = None


def run_selection_tracked_training_loop(
    *,
    model: nn.Module,
    epochs: int,
    train_epoch: Callable[[int], SelectionTrackedEpochResult],
    evaluate_selection: Callable[[], dict[str, Any]],
    best_checkpoint_error_message: str,
    log_epoch_summary: Callable[[str], None] | None = None,
    initial_history: Sequence[Mapping[str, Any]] | None = None,
    initial_best_checkpoint_state: dict[str, Any] | None = None,
    after_epoch: Callable[[int, list[dict[str, Any]], dict[str, Any]], None]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """epoch 학습 결과를 selection 평가, history, best checkpoint로 닫는다.

    실제 batch 구성, model input 변환, optimizer step은 caller의 `train_epoch`가
    소유한다. 이 helper는 fixed classifier와 LoRA classifier에서 동일한
    selection-set 기반 검증/기록/복원 흐름만 담당한다.
    """

    history: list[dict[str, Any]] = (
        [] if initial_history is None else [dict(record) for record in initial_history]
    )
    best_checkpoint = BestModelCheckpoint(
        initial_state=initial_best_checkpoint_state,
    )

    initial_epoch_count = len(history)
    for epoch in range(1, epochs + 1):
        epoch_number = initial_epoch_count + epoch
        train_result = train_epoch(epoch_number)
        selection_report = evaluate_selection()
        epoch_record = build_selection_epoch_record(
            epoch=epoch_number,
            train_loss_total=train_result.train_loss_total,
            train_loss_denominator=train_result.train_loss_denominator,
            selection_report=selection_report,
            extra_train_metrics=train_result.extra_train_metrics,
        )
        history.append(epoch_record)
        if log_epoch_summary is not None:
            log_epoch_summary(
                f"[epoch={epoch_number}] {format_selection_epoch_summary(epoch_record)}"
            )
        best_checkpoint.update(model=model, selection_report=selection_report)
        if after_epoch is not None:
            after_epoch(epoch_number, history, best_checkpoint.state_dict())

    best_selection_report = best_checkpoint.restore_best(
        model=model,
        error_message=best_checkpoint_error_message,
    )
    return history, best_selection_report
