"""Method-owned privacy guard registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .base import PrivacyGuardFactory, SharedAdapterPrivacyGuard

_PRIVACY_GUARDS_PACKAGE = "methods.adaptation.privacy_guards"
_SKIPPED_PRIVACY_GUARD_MODULES = frozenset({"base", "registry"})
_PRIVACY_GUARD_REGISTRY: dict[
    str,
    tuple[PrivacyGuardFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_privacy_guard(
    *guard_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: PrivacyGuardFactory | None = None,
) -> Callable[[PrivacyGuardFactory], PrivacyGuardFactory] | PrivacyGuardFactory:
    """privacy guard 구현 옆에서 method-owned wiring을 등록한다."""

    def _decorator(factory: PrivacyGuardFactory) -> PrivacyGuardFactory:
        registered_item = (factory, catalog_entry)
        for guard_name in guard_names:
            _PRIVACY_GUARD_REGISTRY[guard_name.strip().lower()] = registered_item
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_shared_adapter_privacy_guard(
    guard_name: str,
) -> SharedAdapterPrivacyGuard:
    """guard 이름으로 method-owned privacy guard를 생성한다."""

    _import_privacy_guard_modules()
    normalized_name = guard_name.strip().lower()
    registered_item = _PRIVACY_GUARD_REGISTRY.get(normalized_name)
    if registered_item is not None:
        factory, _catalog_entry = registered_item
        return factory()
    raise ValueError(f"Unsupported privacy guard: {guard_name}.")


def list_registered_shared_adapter_privacy_guard_names() -> tuple[str, ...]:
    """등록된 privacy guard 이름을 정렬된 tuple로 반환한다."""

    _import_privacy_guard_modules()
    return tuple(sorted(_PRIVACY_GUARD_REGISTRY))


def list_shared_adapter_privacy_guard_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 privacy guard catalog entry를 canonical item 기준으로 반환한다."""

    _import_privacy_guard_modules()
    return dedupe_registry_catalog_entries(
        catalog_entry for _factory, catalog_entry in _PRIVACY_GUARD_REGISTRY.values()
    )


def _import_privacy_guard_modules() -> None:
    package = importlib.import_module(_PRIVACY_GUARDS_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.iter_modules(package_paths):
        if module_info.name in _SKIPPED_PRIVACY_GUARD_MODULES:
            continue
        importlib.import_module(f"{_PRIVACY_GUARDS_PACKAGE}.{module_info.name}")
