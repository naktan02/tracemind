"""Top1-based acceptance policy metadata."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry

from .registry import register_pseudo_label_acceptance_policy

TOP1_MARGIN_THRESHOLD_ACCEPTANCE_POLICY_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="top1_margin_threshold",
    display_name="top1_margin_threshold",
    implementation_module="agent.src.services.training.acceptance_policies.top1",
    core_method_name="top1_margin_threshold",
    family_name="pseudo_label_acceptance",
    supported_adapter_kinds=("*",),
)
TOP1_CONFIDENCE_ONLY_ACCEPTANCE_POLICY_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="top1_confidence_only",
    display_name="top1_confidence_only",
    implementation_module="agent.src.services.training.acceptance_policies.top1",
    core_method_name="top1_confidence_only",
    family_name="pseudo_label_acceptance",
    supported_adapter_kinds=("*",),
)


@register_pseudo_label_acceptance_policy(
    "top1_margin_threshold",
    catalog_entry=TOP1_MARGIN_THRESHOLD_ACCEPTANCE_POLICY_CATALOG_ENTRY,
)
@dataclass(slots=True)
class Top1MarginThresholdAcceptancePolicy:
    """Top1 confidence와 margin selection hook의 runtime compatibility alias."""

    policy_name: str = "top1_margin_threshold"
    selection_hook_name: str = "top1_margin_threshold"
    supported_adapter_kinds: tuple[str, ...] = ("*",)


@register_pseudo_label_acceptance_policy(
    "top1_confidence_only",
    catalog_entry=TOP1_CONFIDENCE_ONLY_ACCEPTANCE_POLICY_CATALOG_ENTRY,
)
@dataclass(slots=True)
class Top1ConfidenceOnlyAcceptancePolicy:
    """Top1 confidence-only selection hook의 runtime compatibility alias."""

    policy_name: str = "top1_confidence_only"
    selection_hook_name: str = "top1_confidence_only"
    supported_adapter_kinds: tuple[str, ...] = ("*",)
