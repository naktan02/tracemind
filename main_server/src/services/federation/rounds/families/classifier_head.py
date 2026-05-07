"""Classifier-head round family."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from shared.src.contracts.adapter_contracts import (
    ClassifierHeadAdapterStatePayload,
    ClassifierHeadAdapterUpdatePayload,
    ClassifierHeadState,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_family_metadata import CLASSIFIER_HEAD_FAMILY_METADATA
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from ..aggregation.classifier_head import (
    ClassifierHeadFedAvgAggregationService,
)
from ..aggregation.diagonal_scale_defaults import AggregationConfigScalar
from ..aggregation.models import SharedAdapterAggregationBackend
from ..aggregation.registry import build_shared_adapter_aggregation_backend
from .models import SharedAdapterRoundFamily
from .registry import register_shared_adapter_round_family


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
                f"State adapter_kind does not match family: {state.adapter_kind}"
            )
        return state


@register_shared_adapter_round_family(CLASSIFIER_HEAD_FAMILY_METADATA.family_name)
def build_classifier_head_round_family(
    aggregation_backend_name: str,
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] | None,
) -> SharedAdapterRoundFamily:
    """classifier-head round family를 server runtime config에서 조립한다."""

    return ClassifierHeadRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
            backend_name=aggregation_backend_name,
            overrides=aggregation_backend_overrides,
        )
    )
