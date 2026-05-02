"""Acceptance policy registry."""

from __future__ import annotations

from collections.abc import Callable

from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .base import PseudoLabelAcceptancePolicy
from .top1 import (
    Top1ConfidenceOnlyAcceptancePolicy,
    Top1MarginThresholdAcceptancePolicy,
)

AcceptancePolicyFactory = Callable[[], PseudoLabelAcceptancePolicy]

_ACCEPTANCE_POLICY_REGISTRY: dict[
    str,
    tuple[AcceptancePolicyFactory, RegistryCatalogEntry],
] = {}


def register_pseudo_label_acceptance_policy(
    *policy_names: str,
    factory: AcceptancePolicyFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """얇은 wiring registry에 pseudo-label acceptance policy를 등록한다."""

    registered_policy = (factory, catalog_entry)
    for policy_name in policy_names:
        _ACCEPTANCE_POLICY_REGISTRY[
            policy_name.strip().lower()
        ] = registered_policy


def build_pseudo_label_acceptance_policy(
    policy_name: str,
) -> PseudoLabelAcceptancePolicy:
    """정책 이름으로 acceptance policy를 생성한다."""

    normalized_name = policy_name.strip().lower()
    registered_policy = _ACCEPTANCE_POLICY_REGISTRY.get(normalized_name)
    if registered_policy is not None:
        factory, _catalog_entry = registered_policy
        return factory()
    raise ValueError(f"Unsupported pseudo-label acceptance policy: {policy_name}.")


def list_registered_pseudo_label_acceptance_policy_names() -> tuple[str, ...]:
    """등록된 acceptance policy 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_ACCEPTANCE_POLICY_REGISTRY))


def list_pseudo_label_acceptance_policy_catalog_entries(
) -> tuple[RegistryCatalogEntry, ...]:
    """등록된 acceptance policy catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _ACCEPTANCE_POLICY_REGISTRY.values()
    )


register_pseudo_label_acceptance_policy(
    "top1_margin_threshold",
    factory=Top1MarginThresholdAcceptancePolicy,
    catalog_entry=RegistryCatalogEntry(
        item_name="top1_margin_threshold",
        display_name="top1_margin_threshold",
        implementation_module=Top1MarginThresholdAcceptancePolicy.__module__,
        core_method_name="top1_margin_threshold",
        family_name="pseudo_label_acceptance",
        supported_adapter_kinds=("*",),
    ),
)
register_pseudo_label_acceptance_policy(
    "top1_confidence_only",
    factory=Top1ConfidenceOnlyAcceptancePolicy,
    catalog_entry=RegistryCatalogEntry(
        item_name="top1_confidence_only",
        display_name="top1_confidence_only",
        implementation_module=Top1ConfidenceOnlyAcceptancePolicy.__module__,
        core_method_name="top1_confidence_only",
        family_name="pseudo_label_acceptance",
        supported_adapter_kinds=("*",),
    ),
)


__all__ = [
    "AcceptancePolicyFactory",
    "build_pseudo_label_acceptance_policy",
    "list_pseudo_label_acceptance_policy_catalog_entries",
    "list_registered_pseudo_label_acceptance_policy_names",
    "register_pseudo_label_acceptance_policy",
]
