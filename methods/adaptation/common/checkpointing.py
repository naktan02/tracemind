"""분류 학습 loop의 best checkpoint 선택 helper."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class BestModelCheckpoint:
    """selection accuracy 기준으로 best model state를 보관한다."""

    def __init__(self, *, metric_key: str = "accuracy_top_1") -> None:
        self._metric_key = metric_key
        self._best_metric = -1.0
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._best_report: dict[str, Any] | None = None

    @property
    def best_report(self) -> dict[str, Any] | None:
        return self._best_report

    def update(
        self,
        *,
        model: nn.Module,
        selection_report: dict[str, Any],
    ) -> None:
        metric = float(selection_report[self._metric_key])
        if metric > self._best_metric:
            self._best_metric = metric
            self._best_state_dict = copy.deepcopy(model.state_dict())
            self._best_report = selection_report

    def restore_best(
        self,
        *,
        model: nn.Module,
        error_message: str,
    ) -> dict[str, Any]:
        if self._best_state_dict is None or self._best_report is None:
            raise RuntimeError(error_message)
        model.load_state_dict(self._best_state_dict)
        return self._best_report
