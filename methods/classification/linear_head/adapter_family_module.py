"""Linear-head shared adapter contract alias."""

from __future__ import annotations

from methods.adaptation.adapter_family_modules import (
    register_adapter_family_module_root,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    CLASSIFIER_HEAD_ADAPTER_KIND,
)

register_adapter_family_module_root(
    CLASSIFIER_HEAD_ADAPTER_KIND,
    module_root="methods.classification.linear_head",
)
