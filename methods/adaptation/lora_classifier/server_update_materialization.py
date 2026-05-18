"""Compatibility entrypoint for LoRA-classifier materialization preflight."""

from __future__ import annotations

from methods.adaptation.lora_classifier import server_preflight as _preflight

AGENT_LOCAL_ARTIFACT_REF_PREFIX = _preflight.AGENT_LOCAL_ARTIFACT_REF_PREFIX
require_lora_classifier_update_is_server_materializable = (
    _preflight.require_lora_classifier_update_is_server_materializable
)
