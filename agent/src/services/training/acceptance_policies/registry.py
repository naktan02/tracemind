"""Acceptance policy registry."""

from __future__ import annotations

from collections.abc import Callable

from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .base import PseudoLabelAcceptancePolicy

AcceptancePolicyFactory = Callable[[], PseudoLabelAcceptancePolicy]

_ACCEPTANCE_POLICY_REGISTRY: dict[
    str,
    tuple[AcceptancePolicyFactory, RegistryCatalogEntry],
] = {}


def register_pseudo_label_acceptance_policy(
    *policy_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: AcceptancePolicyFactory | None = None,
) -> (
    Callable[[AcceptancePolicyFactory], AcceptancePolicyFactory]
    | AcceptancePolicyFactory
):
    """acceptance policy metadata factory 옆에서 runtime wiring을 등록한다."""

    def _decorator(factory: AcceptancePolicyFactory) -> AcceptancePolicyFactory:
        registered_policy = (factory, catalog_entry)
        for policy_name in policy_names:
            _ACCEPTANCE_POLICY_REGISTRY[policy_name.strip().lower()] = (
                registered_policy
            )
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_pseudo_label_acceptance_policy(
    policy_name: str,
) -> PseudoLabelAcceptancePolicy:
    """정책 이름으로 acceptance policy를 생성한다."""

    _ensure_builtin_pseudo_label_acceptance_policies_loaded()
    normalized_name = policy_name.strip().lower()
    registered_policy = _ACCEPTANCE_POLICY_REGISTRY.get(normalized_name)
    if registered_policy is not None:
        factory, _catalog_entry = registered_policy
        return factory()
    raise ValueError(f"Unsupported pseudo-label acceptance policy: {policy_name}.")


def list_registered_pseudo_label_acceptance_policy_names() -> tuple[str, ...]:
    """등록된 acceptance policy 이름을 정렬된 tuple로 반환한다."""

    _ensure_builtin_pseudo_label_acceptance_policies_loaded()
    return tuple(sorted(_ACCEPTANCE_POLICY_REGISTRY))


def list_pseudo_label_acceptance_policy_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 acceptance policy catalog entry를 canonical item 기준으로 반환한다."""

    _ensure_builtin_pseudo_label_acceptance_policies_loaded()
    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _ACCEPTANCE_POLICY_REGISTRY.values()
    )


def _ensure_builtin_pseudo_label_acceptance_policies_loaded() -> None:
    from agent.src.services.training.acceptance_policies.builtin_loader import (
        load_builtin_pseudo_label_acceptance_policies,
    )

    load_builtin_pseudo_label_acceptance_policies()
