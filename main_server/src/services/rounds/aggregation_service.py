"""Shared adapter 집계 서비스.

현재 concrete 구현은 diagonal scale adapter 하나만 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence

from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)
from shared.src.domain.entities.training.vector_adapter_delta import (
    VectorAdapterDelta,
)
from shared.src.domain.entities.training.vector_adapter_state import (
    VectorAdapterState,
)


@dataclass(slots=True)
class AggregationConfig:
    """전역 adapter state 집계 시 적용할 안전 범위."""

    min_scale: float = 0.75
    max_scale: float = 1.25


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


@dataclass(slots=True)
class DiagonalScaleAggregationService:
    """Diagonal scale adapter update를 전역 상태로 집계한다."""

    adapter_kind: str = "diagonal_scale"
    config: AggregationConfig = field(default_factory=AggregationConfig)

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, VectorAdapterState):
            raise TypeError(
                "DiagonalScaleAggregationService expects VectorAdapterState as the "
                f"base state, got {type(base_state)!r}."
            )
        if base_state.adapter_kind != self.adapter_kind:
            raise ValueError(
                "Base state adapter_kind does not match the diagonal scale "
                f"aggregator: {base_state.adapter_kind}"
            )

        valid_updates = [
            payload for payload in update_payloads if payload.example_count > 0
        ]
        if not valid_updates:
            raise ValueError("At least one non-empty update payload is required.")

        embedding_dim = base_state.embedding_dim
        total_examples = sum(payload.example_count for payload in valid_updates)
        weighted_delta = [0.0] * embedding_dim
        weighted_confidence = 0.0
        weighted_margin = 0.0
        weighted_delta_norm = 0.0

        for payload in valid_updates:
            if not isinstance(payload, VectorAdapterDelta):
                raise TypeError(
                    "DiagonalScaleAggregationService expects VectorAdapterDelta "
                    f"updates, got {type(payload)!r}."
                )
            if payload.adapter_kind != self.adapter_kind:
                raise ValueError(
                    "Update adapter_kind does not match the diagonal scale "
                    f"aggregator: {payload.adapter_kind}"
                )
            if payload.model_id != base_state.model_id:
                raise ValueError("All update payloads must match the base model_id.")
            if payload.base_model_revision != base_state.model_revision:
                raise ValueError(
                    "All update payloads must match the base model revision."
                )
            if payload.training_scope != base_state.training_scope:
                raise ValueError("All update payloads must match the training scope.")
            if payload.embedding_dim != embedding_dim:
                raise ValueError(
                    "All update payloads must share the same embedding_dim."
                )

            weight = payload.example_count / total_examples
            weighted_confidence += payload.mean_confidence * weight
            weighted_margin += (payload.mean_margin or 0.0) * weight
            weighted_delta_norm += payload.l2_norm() * weight
            for index, value in enumerate(payload.dimension_deltas):
                weighted_delta[index] += value * weight

        next_scales = [
            max(
                self.config.min_scale,
                min(self.config.max_scale, scale + delta),
            )
            for scale, delta in zip(
                base_state.dimension_scales,
                weighted_delta,
                strict=True,
            )
        ]
        next_state = VectorAdapterState(
            schema_version=base_state.schema_version,
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            dimension_scales=next_scales,
            updated_at=aggregated_at,
            adapter_kind=base_state.adapter_kind,
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics={
                "client_count": float(len(valid_updates)),
                "example_count": float(total_examples),
                "mean_confidence": weighted_confidence,
                "mean_margin": weighted_margin,
                "mean_delta_l2_norm": weighted_delta_norm,
            },
            update_count=len(valid_updates),
        )


AggregationService = DiagonalScaleAggregationService
