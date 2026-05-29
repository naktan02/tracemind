"""Linear-head payload adapter module manifest."""

from __future__ import annotations

from methods.adaptation.implementation_modules import (
    register_adaptation_implementation_module_root,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_ADAPTER_KIND,
)

register_adaptation_implementation_module_root(
    CLASSIFIER_HEAD_ADAPTER_KIND,
    module_root="methods.classification.linear_head",
)
