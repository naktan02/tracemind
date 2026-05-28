"""PEFT text encoder payload adapter module manifest."""

from __future__ import annotations

from methods.adaptation.payload_adapter_modules import (
    register_payload_adapter_module_root,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
)

register_payload_adapter_module_root(
    PEFT_CLASSIFIER_ADAPTER_KIND,
    module_root="methods.adaptation.peft_text_encoder",
)
