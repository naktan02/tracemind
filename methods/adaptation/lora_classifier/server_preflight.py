"""Compatibility shim for legacy lora_classifier server preflight imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.server_preflight import (
    AGENT_LOCAL_ARTIFACT_REF_PREFIX,
    require_lora_classifier_update_is_server_materializable,
    require_lora_classifier_update_matches_active_state,
)
