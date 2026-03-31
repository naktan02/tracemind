"""Shared adapter family별 서버 라운드 조합 객체."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from shared.src.contracts.adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
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
from main_server.src.services.rounds.aggregation_service import (
    DiagonalScaleAggregationService,
    SharedAdapterAggregationBackend,
)


class SharedAdapterRoundFamily(Protocol):
    """adapter family별 payload 변환과 집계 backend 조합."""

    adapter_kind: str
    accepted_update_formats: tuple[str, ...]
    aggregation_backend: SharedAdapterAggregationBackend

    def state_from_payload(
        self,
        payload: SharedAdapterStatePayload,
    ) -> SharedAdapterState:
        """contract payload를 domain state로 변환한다."""

    def update_from_payload(
        self,
        payload: SharedAdapterUpdatePayload,
    ) -> SharedAdapterUpdate:
        """contract payload를 domain update로 변환한다."""

    def state_to_payload(
        self,
        state: SharedAdapterState,
    ) -> SharedAdapterStatePayload:
        """domain state를 contract payload로 변환한다."""


@dataclass(slots=True)
class DiagonalScaleRoundFamily:
    """현재 diagonal scale adapter family의 서버 측 조합 객체."""

    adapter_kind: str = "diagonal_scale"
    accepted_update_formats: tuple[str, ...] = (
        "diagonal_scale_update",
        "vector_adapter_delta",
    )
    aggregation_backend: SharedAdapterAggregationBackend = field(
        default_factory=DiagonalScaleAggregationService
    )

    def state_from_payload(
        self,
        payload: SharedAdapterStatePayload,
    ) -> SharedAdapterState:
        if not isinstance(payload, DiagonalScaleAdapterStatePayload):
            raise TypeError(
                "DiagonalScaleRoundFamily expects "
                "DiagonalScaleAdapterStatePayload, "
                f"got {type(payload)!r}."
            )
        if payload.adapter_kind != self.adapter_kind:
            raise ValueError(
                "State payload adapter_kind does not match family: "
                f"{payload.adapter_kind}"
            )
        return VectorAdapterState(
            schema_version=payload.schema_version,
            model_id=payload.model_id,
            model_revision=payload.model_revision,
            training_scope=payload.training_scope,
            dimension_scales=list(payload.dimension_scales),
            updated_at=payload.updated_at,
            adapter_kind=payload.adapter_kind,
        )

    def update_from_payload(
        self,
        payload: SharedAdapterUpdatePayload,
    ) -> SharedAdapterUpdate:
        if not isinstance(payload, DiagonalScaleAdapterUpdatePayload):
            raise TypeError(
                "DiagonalScaleRoundFamily expects "
                "DiagonalScaleAdapterUpdatePayload, "
                f"got {type(payload)!r}."
            )
        if payload.adapter_kind != self.adapter_kind:
            raise ValueError(
                "Update payload adapter_kind does not match family: "
                f"{payload.adapter_kind}"
            )
        return VectorAdapterDelta(
            schema_version=payload.schema_version,
            model_id=payload.model_id,
            base_model_revision=payload.base_model_revision,
            training_scope=payload.training_scope,
            dimension_deltas=list(payload.dimension_deltas),
            example_count=payload.example_count,
            mean_confidence=payload.mean_confidence,
            created_at=payload.created_at,
            mean_margin=payload.mean_margin,
            label_counts=dict(payload.label_counts),
            adapter_kind=payload.adapter_kind,
        )

    def state_to_payload(
        self,
        state: SharedAdapterState,
    ) -> SharedAdapterStatePayload:
        if not isinstance(state, VectorAdapterState):
            raise TypeError(
                "DiagonalScaleRoundFamily expects VectorAdapterState, "
                f"got {type(state)!r}."
            )
        if state.adapter_kind != self.adapter_kind:
            raise ValueError(
                "State adapter_kind does not match family: "
                f"{state.adapter_kind}"
            )
        return DiagonalScaleAdapterStatePayload(
            schema_version=state.schema_version,
            adapter_kind=state.adapter_kind,
            model_id=state.model_id,
            model_revision=state.model_revision,
            training_scope=state.training_scope,
            dimension_scales=state.dimension_scales,
            updated_at=state.updated_at,
        )
