"""Server-side update materialization preflight."""

from __future__ import annotations

from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
)

AGENT_LOCAL_ARTIFACT_REF_PREFIX = "agent-local://"


def require_server_materializable_update_payload(
    update_payload: SharedAdapterUpdatePayload,
) -> None:
    """서버가 finalize 전에 materialize할 수 없는 update payload를 거부한다."""

    if isinstance(update_payload, LoraClassifierDelta):
        _require_lora_classifier_update_is_server_materializable(update_payload)


def _require_lora_classifier_update_is_server_materializable(
    update_payload: LoraClassifierDelta,
) -> None:
    lora_ref_required = update_payload.lora_parameter_deltas is None
    head_ref_required = update_payload.classifier_head_weight_deltas is None
    unsupported_refs = [
        artifact_ref
        for artifact_ref in (
            update_payload.lora_delta_artifact_ref if lora_ref_required else None,
            (
                update_payload.classifier_head_delta_artifact_ref
                if head_ref_required
                else None
            ),
        )
        if artifact_ref is not None
        and artifact_ref.startswith(AGENT_LOCAL_ARTIFACT_REF_PREFIX)
    ]
    if unsupported_refs:
        raise ValueError(
            "LoRA-classifier update uses agent-local artifact ref(s) that the "
            "server cannot materialize yet: "
            f"{unsupported_refs}. Upload/materialize them as server-owned refs "
            "or send inline deltas."
        )
