"""Shared adapter family별 서버 라운드 조합 객체."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from main_server.src.services.rounds.aggregation_service import (
    ClassifierHeadFedAvgAggregationService,
    DiagonalScaleAggregationService,
    SharedAdapterAggregationBackend,
    build_shared_adapter_aggregation_backend,
)
from main_server.src.services.rounds.diagonal_scale_defaults import (
    AggregationConfigScalar,
)
from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadAdapterStatePayload,
    ClassifierHeadAdapterUpdatePayload,
    ClassifierHeadState,
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


RoundFamilyFactory = Callable[
    [str, Mapping[str, AggregationConfigScalar] | None],
    SharedAdapterRoundFamily,
]


@dataclass(slots=True)
class DiagonalScaleRoundFamily:
    """현재 diagonal scale adapter family의 서버 측 조합 객체."""

    adapter_kind: str = DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind
    accepted_update_formats: tuple[str, ...] = (
        DIAGONAL_SCALE_FAMILY_METADATA.accepted_update_payload_formats
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


@dataclass(slots=True)
class ClassifierHeadRoundFamily:
    """Classifier-head family의 서버 측 조합 객체."""

    adapter_kind: str = CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind
    accepted_update_formats: tuple[str, ...] = (
        CLASSIFIER_HEAD_FAMILY_METADATA.accepted_update_payload_formats
    )
    aggregation_backend: SharedAdapterAggregationBackend = field(
        default_factory=ClassifierHeadFedAvgAggregationService
    )

    def state_from_payload(
        self,
        payload: SharedAdapterStatePayload,
    ) -> SharedAdapterState:
        if not isinstance(payload, ClassifierHeadAdapterStatePayload):
            raise TypeError(
                "ClassifierHeadRoundFamily expects "
                "ClassifierHeadAdapterStatePayload, "
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
        if not isinstance(payload, ClassifierHeadAdapterUpdatePayload):
            raise TypeError(
                "ClassifierHeadRoundFamily expects "
                "ClassifierHeadAdapterUpdatePayload, "
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
        if not isinstance(state, ClassifierHeadState):
            raise TypeError(
                "ClassifierHeadRoundFamily expects ClassifierHeadState, "
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
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None = None,
) -> SharedAdapterRoundFamily:
    """adapter family와 aggregation backend 이름으로 서버 조합 객체를 만든다."""

    normalized_family_name = family_name.strip().lower()
    factory = _ROUND_FAMILY_REGISTRY.get(normalized_family_name)
    if factory is not None:
        return factory(aggregation_backend_name, aggregation_backend_overrides)
    raise ValueError(f"Unsupported shared adapter family: {family_name}.")


def _build_diagonal_scale_round_family(
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None,
) -> SharedAdapterRoundFamily:
    return DiagonalScaleRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        )
    )


def _build_classifier_head_round_family(
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None,
) -> SharedAdapterRoundFamily:
    return ClassifierHeadRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        )
    )


register_shared_adapter_round_family(
    DIAGONAL_SCALE_FAMILY_METADATA.family_name,
    factory=_build_diagonal_scale_round_family,
)
register_shared_adapter_round_family(
    CLASSIFIER_HEAD_FAMILY_METADATA.family_name,
    factory=_build_classifier_head_round_family,
)
