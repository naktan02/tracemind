"""Acceptance policy registry."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry import RuntimeRegistry
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry

from .base import PseudoLabelAcceptancePolicy

AcceptancePolicyFactory = Callable[[], PseudoLabelAcceptancePolicy]

_ACCEPTANCE_POLICY_REGISTRY = RuntimeRegistry[AcceptancePolicyFactory](
    package_name="agent.src.services.training.acceptance_policies",
    item_kind="pseudo-label acceptance policy",
)


def register_pseudo_label_acceptance_policy(
    *policy_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: AcceptancePolicyFactory | None = None,
) -> (
    Callable[[AcceptancePolicyFactory], AcceptancePolicyFactory]
    | AcceptancePolicyFactory
):
    """acceptance policy metadata factory 옆에서 runtime wiring을 등록한다."""

    return _ACCEPTANCE_POLICY_REGISTRY.register(
        *policy_names,
        catalog_entry=catalog_entry,
        factory=factory,
    )


def build_pseudo_label_acceptance_policy(
    policy_name: str,
) -> PseudoLabelAcceptancePolicy:
    """정책 이름으로 acceptance policy를 생성한다."""

    factory, _catalog_entry = _ACCEPTANCE_POLICY_REGISTRY.get(policy_name)
    return factory()


def list_registered_pseudo_label_acceptance_policy_names() -> tuple[str, ...]:
    """등록된 acceptance policy 이름을 정렬된 tuple로 반환한다."""

    return _ACCEPTANCE_POLICY_REGISTRY.list_names()


def list_pseudo_label_acceptance_policy_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 acceptance policy catalog entry를 canonical item 기준으로 반환한다."""

    return _ACCEPTANCE_POLICY_REGISTRY.list_catalog_entries()
