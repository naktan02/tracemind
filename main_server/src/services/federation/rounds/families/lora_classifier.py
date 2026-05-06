"""LoRA-classifier round family."""

from __future__ import annotations

from dataclasses import dataclass, field

from main_server.src.services.federation.rounds.aggregation.lora_classifier import (
    LoraClassifierFedAvgAggregationService,
)
from main_server.src.services.federation.rounds.aggregation.models import (
    SharedAdapterAggregationBackend,
)
from shared.src.config.adapter_family_metadata import LORA_CLASSIFIER_FAMILY_METADATA
from shared.src.contracts.adapter_contracts import (
    LoraClassifierAdapterStatePayload,
    LoraClassifierAdapterUpdatePayload,
    LoraClassifierState,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


@dataclass(slots=True)
class LoraClassifierRoundFamily:
    """LoRA-classifier family의 서버 측 조합 객체."""

    adapter_kind: str = LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind
    accepted_update_formats: tuple[str, ...] = (
        LORA_CLASSIFIER_FAMILY_METADATA.accepted_update_payload_formats
    )
    aggregation_backend: SharedAdapterAggregationBackend = field(
        default_factory=LoraClassifierFedAvgAggregationService
    )

    def state_from_payload(
        self,
        payload: SharedAdapterStatePayload,
    ) -> SharedAdapterState:
        if not isinstance(payload, LoraClassifierAdapterStatePayload):
            raise TypeError(
                "LoraClassifierRoundFamily expects "
                "LoraClassifierAdapterStatePayload, "
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
        if not isinstance(payload, LoraClassifierAdapterUpdatePayload):
            raise TypeError(
                "LoraClassifierRoundFamily expects "
                "LoraClassifierAdapterUpdatePayload, "
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
        if not isinstance(state, LoraClassifierState):
            raise TypeError(
                "LoraClassifierRoundFamily expects LoraClassifierState, "
                f"got {type(state)!r}."
            )
        if state.adapter_kind != self.adapter_kind:
            raise ValueError(
                f"State adapter_kind does not match family: {state.adapter_kind}"
            )
        return state
