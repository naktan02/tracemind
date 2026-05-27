"""LoRA-classifier training scalar metric accumulation primitive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from shared.src.domain.services.classification_report import safe_divide


@dataclass(slots=True)
class ScalarMetricAccumulator:
    """scalar tensor/float 값을 누적하고 평균 record로 변환한다."""

    sums: dict[str, float] = field(default_factory=dict)

    def add_float(self, name: str, value: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value)

    def add_tensor(self, name: str, value: Tensor) -> None:
        self.add_float(name, float(value.detach().item()))

    def add_tensor_mapping(
        self,
        values: Mapping[str, Tensor],
        *,
        prefix: str = "",
    ) -> None:
        for name, value in values.items():
            self.add_tensor(f"{prefix}{name}", value)

    def average_record(
        self,
        *,
        denominator: int,
        key_prefix: str = "",
    ) -> dict[str, float]:
        return {
            f"{key_prefix}{name}": round(safe_divide(total, denominator), 6)
            for name, total in self.sums.items()
        }

    def running_fields(
        self,
        *,
        denominator: int,
        key_prefix: str = "",
    ) -> list[str]:
        return [
            f"{key_prefix}{name}={safe_divide(total, denominator):.4f}"
            for name, total in self.sums.items()
        ]


def tensor_mapping_l2(parameter_deltas: Mapping[str, Tensor]) -> float:
    """tensor mapping의 L2 norm을 float로 계산한다."""

    squared_norm = 0.0
    for delta in parameter_deltas.values():
        squared_norm += float(torch.sum(torch.square(delta.detach())).item())
    return squared_norm**0.5
