"""Privacy guard registry primitive."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry import RuntimeRegistry
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry

from .base import PrivacyGuardFactory, SharedAdapterPrivacyGuard

_PRIVACY_GUARD_REGISTRY = RuntimeRegistry[PrivacyGuardFactory](
    package_name="agent.src.services.training.execution.privacy_guards",
    item_kind="privacy guard",
)


def register_shared_adapter_privacy_guard(
    *guard_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: PrivacyGuardFactory | None = None,
) -> Callable[[PrivacyGuardFactory], PrivacyGuardFactory] | PrivacyGuardFactory:
    """privacy guard 구현 옆에서 runtime wiring을 등록한다."""

    return _PRIVACY_GUARD_REGISTRY.register(
        *guard_names,
        catalog_entry=catalog_entry,
        factory=factory,
    )


def build_shared_adapter_privacy_guard(
    guard_name: str,
) -> SharedAdapterPrivacyGuard:
    """guard 이름으로 privacy guard를 생성한다."""

    factory, _catalog_entry = _PRIVACY_GUARD_REGISTRY.get(guard_name)
    return factory()


def list_registered_shared_adapter_privacy_guard_names() -> tuple[str, ...]:
    """등록된 privacy guard 이름을 정렬된 tuple로 반환한다."""

    return _PRIVACY_GUARD_REGISTRY.list_names()


def list_shared_adapter_privacy_guard_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 privacy guard catalog entry를 canonical item 기준으로 반환한다."""

    return _PRIVACY_GUARD_REGISTRY.list_catalog_entries()
