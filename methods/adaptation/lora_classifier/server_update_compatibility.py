"""Compatibility entrypoint for LoRA-classifier update/state preflight."""

from __future__ import annotations

from methods.adaptation.lora_classifier import server_preflight as _preflight

require_lora_classifier_update_matches_active_state = (
    _preflight.require_lora_classifier_update_matches_active_state
)
