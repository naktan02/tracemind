"""분류 학습 loop의 best checkpoint 선택 helper."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class BestModelCheckpoint:
    """selection accuracy 기준으로 best model state를 보관한다."""

    def __init__(
        self,
        *,
        metric_key: str = "accuracy_top_1",
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self._metric_key = metric_key
        self._best_metric = -1.0
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._best_report: dict[str, Any] | None = None
        if initial_state is not None:
            self.load_state_dict(initial_state)

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

    def state_dict(self) -> dict[str, Any]:
        """resume checkpoint에 저장할 best-selection state."""

        return {
            "metric_key": self._metric_key,
            "best_metric": self._best_metric,
            "best_state_dict": copy.deepcopy(self._best_state_dict),
            "best_report": copy.deepcopy(self._best_report),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """resume checkpoint에서 best-selection state를 복원한다."""

        metric_key = str(state.get("metric_key", self._metric_key))
        if metric_key != self._metric_key:
            raise ValueError("Best checkpoint metric_key does not match.")
        self._best_metric = float(state.get("best_metric", -1.0))
        best_state_dict = state.get("best_state_dict")
        self._best_state_dict = (
            None if best_state_dict is None else copy.deepcopy(best_state_dict)
        )
        best_report = state.get("best_report")
        self._best_report = None if best_report is None else copy.deepcopy(best_report)
