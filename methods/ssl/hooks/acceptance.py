"""Pseudo-label acceptance policy specs.

Acceptance policy 이름은 SSL selection hook 의미에 붙는 method-level spec이다.
agent는 이 spec을 검증에만 사용하고, local candidate/context 조립만 맡는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry

from .registry import register_pseudo_label_acceptance_policy

ANY_ADAPTER_KIND = "*"


@dataclass(frozen=True, slots=True)
class PseudoLabelAcceptancePolicySpec:
    """runtime compatibility가 보는 pseudo-label acceptance policy spec."""

    policy_name: str
    selection_hook_name: str
    supported_adapter_kinds: tuple[str, ...] = (ANY_ADAPTER_KIND,)


TOP1_MARGIN_THRESHOLD_ACCEPTANCE_POLICY_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="top1_margin_threshold",
    display_name="top1_margin_threshold",
    implementation_module="methods.ssl.hooks.acceptance",
    core_method_name="top1_margin_threshold",
    family_name="pseudo_label_acceptance",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)
TOP1_CONFIDENCE_ONLY_ACCEPTANCE_POLICY_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="top1_confidence_only",
    display_name="top1_confidence_only",
    implementation_module="methods.ssl.hooks.acceptance",
    core_method_name="top1_confidence_only",
    family_name="pseudo_label_acceptance",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)
TOP1_RANKED_ACCEPTANCE_POLICY_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="top1_ranked",
    display_name="top1_ranked",
    implementation_module="methods.ssl.hooks.acceptance",
    core_method_name="top1_ranked",
    family_name="pseudo_label_acceptance",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)


@register_pseudo_label_acceptance_policy(
    "top1_margin_threshold",
    catalog_entry=TOP1_MARGIN_THRESHOLD_ACCEPTANCE_POLICY_CATALOG_ENTRY,
)
def build_top1_margin_threshold_acceptance_policy() -> PseudoLabelAcceptancePolicySpec:
    """Top1 confidence와 margin selection hook을 acceptance spec으로 노출한다."""

    return PseudoLabelAcceptancePolicySpec(
        policy_name="top1_margin_threshold",
        selection_hook_name="top1_margin_threshold",
    )


@register_pseudo_label_acceptance_policy(
    "top1_confidence_only",
    catalog_entry=TOP1_CONFIDENCE_ONLY_ACCEPTANCE_POLICY_CATALOG_ENTRY,
)
def build_top1_confidence_only_acceptance_policy() -> PseudoLabelAcceptancePolicySpec:
    """Top1 confidence-only selection hook을 acceptance spec으로 노출한다."""

    return PseudoLabelAcceptancePolicySpec(
        policy_name="top1_confidence_only",
        selection_hook_name="top1_confidence_only",
    )


@register_pseudo_label_acceptance_policy(
    "top1_ranked",
    catalog_entry=TOP1_RANKED_ACCEPTANCE_POLICY_CATALOG_ENTRY,
)
def build_top1_ranked_acceptance_policy() -> PseudoLabelAcceptancePolicySpec:
    """Cutoff 없이 top1 evidence를 ranking/cap 대상으로 여는 acceptance spec."""

    return PseudoLabelAcceptancePolicySpec(
        policy_name="top1_ranked",
        selection_hook_name="top1_ranked",
    )
