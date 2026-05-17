"""LoRA-classifier server-side update/state compatibility preflight."""

from __future__ import annotations

from methods.adaptation.server_update_compatibility import (
    register_server_update_compatibility_validator,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


@register_server_update_compatibility_validator(LORA_CLASSIFIER_ADAPTER_KIND)
def require_lora_classifier_update_matches_active_state(
    update_payload: SharedAdapterUpdatePayload,
    active_state: SharedAdapterState,
) -> None:
    """LoRA-classifier update가 active state/manifest와 같은 family인지 검사한다."""

    if not isinstance(update_payload, LoraClassifierDelta):
        raise ValueError(
            "LoRA-classifier compatibility expects LoraClassifierDelta payload."
        )
    if not isinstance(active_state, LoraClassifierState):
        raise ValueError(
            "LoRA-classifier compatibility expects active LoraClassifierState."
        )

    _require_equal(
        "model_id",
        update_payload.model_id,
        active_state.model_id,
    )
    _require_equal(
        "base_model_revision",
        update_payload.base_model_revision,
        active_state.model_revision,
    )
    _require_equal(
        "training_scope",
        str(update_payload.training_scope),
        str(active_state.training_scope),
    )
    _require_equal(
        "backbone",
        update_payload.backbone.model_dump(mode="json"),
        active_state.backbone.model_dump(mode="json"),
    )
    _require_equal(
        "lora_config",
        update_payload.lora_config.model_dump(mode="json"),
        active_state.lora_config.model_dump(mode="json"),
    )
    _require_equal(
        "label_schema",
        tuple(update_payload.label_schema),
        tuple(active_state.label_schema),
    )


def _require_equal(field_name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            "LoRA-classifier update is not compatible with the active state: "
            f"{field_name} {actual!r} != {expected!r}."
        )
