"""Shared adapter aggregation backend contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .diagonal_scale_defaults import (
    AggregationConfigScalar,
    DiagonalScaleFedAvgAggregationConfig,
)

AggregationConfig = DiagonalScaleFedAvgAggregationConfig


@dataclass(slots=True)
class AggregationResult:
    """집계 결과로 만들어진 새 전역 상태와 메트릭."""

    next_state: SharedAdapterState
    aggregated_metrics: dict[str, float]
    update_count: int


class SharedAdapterAggregationBackend(Protocol):
    """Shared adapter 종류별 서버 집계 backend 인터페이스."""

    adapter_kind: str

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        """같은 adapter family의 update들을 새 전역 상태로 합친다."""


AggregationBackendFactory = Callable[
    [Mapping[str, AggregationConfigScalar] | None],
    SharedAdapterAggregationBackend,
]
