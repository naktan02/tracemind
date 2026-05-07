"""Shared adapter round family contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from shared.src.contracts.adapter_contracts import (
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from ..aggregation.diagonal_scale_defaults import AggregationConfigScalar
from ..aggregation.models import SharedAdapterAggregationBackend


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


RoundFamilyFactory = Callable[
    [str, Mapping[str, AggregationConfigScalar] | None],
    SharedAdapterRoundFamily,
]


@dataclass(slots=True)
class SharedAdapterRoundFamilyRuntime:
    """adapter family contract를 server round runtime에 연결하는 generic 조합 객체."""

    adapter_kind: str
    accepted_update_formats: tuple[str, ...]
    aggregation_backend: SharedAdapterAggregationBackend

    def state_from_payload(
        self,
        payload: SharedAdapterStatePayload,
    ) -> SharedAdapterState:
        """contract state payload가 현재 family와 맞는지 검증한다."""

        if not isinstance(payload, SharedAdapterStatePayload):
            raise TypeError(
                "SharedAdapterRoundFamilyRuntime expects "
                "SharedAdapterStatePayload, "
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
        """contract update payload가 현재 family와 맞는지 검증한다."""

        if not isinstance(payload, SharedAdapterUpdatePayload):
            raise TypeError(
                "SharedAdapterRoundFamilyRuntime expects "
                "SharedAdapterUpdatePayload, "
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
        """aggregation 결과 state를 저장 가능한 contract payload로 검증한다."""

        if not isinstance(state, SharedAdapterStatePayload):
            raise TypeError(
                "SharedAdapterRoundFamilyRuntime expects "
                "SharedAdapterStatePayload, "
                f"got {type(state)!r}."
            )
        if state.adapter_kind != self.adapter_kind:
            raise ValueError(
                f"State adapter_kind does not match family: {state.adapter_kind}"
            )
        return state
