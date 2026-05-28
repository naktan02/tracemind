"""PEFT encoder head server-side update preflight rules."""

from __future__ import annotations

from methods.adaptation.server_update_compatibility import (
    register_server_update_compatibility_validator,
)
from methods.adaptation.server_update_materialization import (
    register_server_update_materialization_validator,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PeftClassifierDelta,
    PeftClassifierState,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

AGENT_LOCAL_ARTIFACT_REF_PREFIX = "agent-local://"
PeftEncoderStatePayload = PeftClassifierState
PeftEncoderDeltaPayload = PeftClassifierDelta


@register_server_update_compatibility_validator(PEFT_CLASSIFIER_ADAPTER_KIND)
def require_peft_encoder_update_matches_active_state(
    update_payload: SharedAdapterUpdatePayload,
    active_state: SharedAdapterState,
) -> None:
    """PEFT encoder head update가 active state/manifest와 맞는지 검사한다."""

    if not isinstance(update_payload, PeftClassifierDelta):
        raise ValueError(
            "PEFT encoder head compatibility expects a PEFT classifier delta payload."
        )
    if not isinstance(active_state, PeftClassifierState):
        raise ValueError(
            "PEFT encoder head compatibility expects active PEFT classifier state."
        )

    _require_equal("model_id", update_payload.model_id, active_state.model_id)
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
        _adapter_config_field_name(update_payload),
        _adapter_config_snapshot(update_payload),
        _adapter_config_snapshot(active_state),
    )
    _require_equal(
        "label_schema",
        tuple(update_payload.label_schema),
        tuple(active_state.label_schema),
    )


@register_server_update_materialization_validator(PEFT_CLASSIFIER_ADAPTER_KIND)
def require_peft_encoder_update_is_server_materializable(
    update_payload: SharedAdapterUpdatePayload,
) -> None:
    """PEFT encoder head update가 서버 local ref만 갖는지 검사한다."""

    if not isinstance(update_payload, PeftClassifierDelta):
        raise ValueError(
            "PEFT encoder head materialization expects a PEFT classifier delta payload."
        )

    peft_ref_required = _peft_parameter_deltas(update_payload) is None
    head_ref_required = update_payload.classifier_head_weight_deltas is None
    unsupported_refs = [
        artifact_ref
        for artifact_ref in (
            _peft_adapter_delta_artifact_ref(update_payload)
            if peft_ref_required
            else None,
            (
                update_payload.classifier_head_delta_artifact_ref
                if head_ref_required
                else None
            ),
            update_payload.partitioned_deltas_artifact_ref,
        )
        if artifact_ref is not None
        and artifact_ref.startswith(AGENT_LOCAL_ARTIFACT_REF_PREFIX)
    ]
    if unsupported_refs:
        raise ValueError(
            "PEFT encoder head update uses agent-local artifact ref(s) that the "
            "server cannot materialize yet: "
            f"{unsupported_refs}. Upload/materialize them as server-owned refs "
            "or send inline deltas."
        )


def _require_equal(field_name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            "PEFT encoder head update is not compatible with the active state: "
            f"{field_name} {actual!r} != {expected!r}."
        )


def _adapter_config_snapshot(
    payload: PeftEncoderStatePayload | PeftEncoderDeltaPayload,
) -> dict[str, object]:
    return payload.peft_adapter_config.model_dump(mode="json")


def _adapter_config_field_name(payload: PeftEncoderDeltaPayload) -> str:
    return "peft_adapter_config"


def _peft_parameter_deltas(
    payload: PeftEncoderDeltaPayload,
) -> dict[str, list[float]] | None:
    return payload.peft_parameter_deltas


def _peft_adapter_delta_artifact_ref(
    payload: PeftEncoderDeltaPayload,
) -> str | None:
    return payload.peft_adapter_delta_artifact_ref
