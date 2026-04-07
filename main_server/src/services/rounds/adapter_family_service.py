"""Shared adapter family별 서버 라운드 조합 객체."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from main_server.src.services.rounds.aggregation_service import (
    DiagonalScaleAggregationService,
    SharedAdapterAggregationBackend,
    build_shared_adapter_aggregation_backend,
)
from shared.src.contracts.adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
    VectorAdapterState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
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


RoundFamilyFactory = Callable[[str], SharedAdapterRoundFamily]


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
        return payload

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
        return payload

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
        return state


_ROUND_FAMILY_REGISTRY: dict[str, RoundFamilyFactory] = {}


def register_shared_adapter_round_family(
    *family_names: str,
    factory: RoundFamilyFactory,
) -> None:
    """adapter family 조합 factory를 얇은 wiring registry에 등록한다."""
    for family_name in family_names:
        _ROUND_FAMILY_REGISTRY[family_name.strip().lower()] = factory


def build_shared_adapter_round_family(
    family_name: str,
    *,
    aggregation_backend_name: str,
) -> SharedAdapterRoundFamily:
    """adapter family와 aggregation backend 이름으로 서버 조합 객체를 만든다."""

    normalized_family_name = family_name.strip().lower()
    factory = _ROUND_FAMILY_REGISTRY.get(normalized_family_name)
    if factory is not None:
        return factory(aggregation_backend_name)
    raise ValueError(f"Unsupported shared adapter family: {family_name}.")


def _build_diagonal_scale_round_family(
    aggregation_backend_name: str,
) -> SharedAdapterRoundFamily:
    return DiagonalScaleRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind="diagonal_scale",
            backend_name=aggregation_backend_name,
        )
    )


register_shared_adapter_round_family(
    "diagonal_scale",
    factory=_build_diagonal_scale_round_family,
)
