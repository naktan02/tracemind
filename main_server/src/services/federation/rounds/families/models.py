"""Shared adapter round family contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
