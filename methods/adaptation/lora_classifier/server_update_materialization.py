"""LoRA-classifier server-side update materialization preflight."""

from __future__ import annotations

from methods.adaptation.server_update_materialization import (
    register_server_update_materialization_validator,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LoraClassifierDelta,
)

AGENT_LOCAL_ARTIFACT_REF_PREFIX = "agent-local://"


@register_server_update_materialization_validator(LORA_CLASSIFIER_ADAPTER_KIND)
def require_lora_classifier_update_is_server_materializable(
    update_payload: SharedAdapterUpdatePayload,
) -> None:
    """LoRA-classifier update가 서버에서 읽을 수 없는 local ref만 갖는지 검사한다."""

    if not isinstance(update_payload, LoraClassifierDelta):
        raise ValueError(
            "LoRA-classifier materialization expects LoraClassifierDelta payload."
        )

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
