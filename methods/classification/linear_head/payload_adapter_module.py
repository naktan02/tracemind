"""Linear-head payload adapter module manifest."""

from __future__ import annotations

from methods.adaptation.payload_adapter_modules import (
    register_payload_adapter_module_root,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_ADAPTER_KIND,
)

register_payload_adapter_module_root(
    CLASSIFIER_HEAD_ADAPTER_KIND,
    module_root="methods.classification.linear_head",
)
