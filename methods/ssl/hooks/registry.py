"""SSL hook registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import TYPE_CHECKING

from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

if TYPE_CHECKING:
    from methods.ssl.hooks.acceptance import PseudoLabelAcceptancePolicySpec
    from methods.ssl.hooks.selection import PseudoLabelSelectionHook

PseudoLabelSelectionHookFactory = Callable[[], "PseudoLabelSelectionHook"]
PseudoLabelAcceptancePolicyFactory = Callable[[], "PseudoLabelAcceptancePolicySpec"]
_SSL_HOOKS_PACKAGE = "methods.ssl.hooks"
_SKIPPED_SSL_HOOK_MODULES = frozenset(
    {
        "base",
        "builtin_loader",
        "objective",
        "registry",
    }
)

_PSEUDO_LABEL_SELECTION_HOOK_REGISTRY: dict[str, PseudoLabelSelectionHookFactory] = {}
_PSEUDO_LABEL_ACCEPTANCE_POLICY_REGISTRY: dict[
    str,
    PseudoLabelAcceptancePolicyFactory,
] = {}
_PSEUDO_LABEL_ACCEPTANCE_POLICY_CATALOG: dict[str, RegistryCatalogEntry] = {}


def register_pseudo_label_selection_hook(
    *hook_names: str,
) -> Callable[[PseudoLabelSelectionHookFactory], PseudoLabelSelectionHookFactory]:
    """이름으로 pseudo-label selection hook을 등록하는 decorator."""

    def _decorator(
        factory: PseudoLabelSelectionHookFactory,
    ) -> PseudoLabelSelectionHookFactory:
        for hook_name in hook_names:
            _PSEUDO_LABEL_SELECTION_HOOK_REGISTRY[hook_name.strip().lower()] = factory
        return factory

    return _decorator


def build_pseudo_label_selection_hook(
    hook_name: str,
) -> PseudoLabelSelectionHook:
    """hook 이름으로 selection hook 구현을 생성한다."""

    _import_ssl_hook_modules()
    normalized_name = hook_name.strip().lower()
    factory = _PSEUDO_LABEL_SELECTION_HOOK_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(f"Unsupported pseudo-label selection hook: {hook_name}.")


def register_pseudo_label_acceptance_policy(
    *policy_names: str,
    catalog_entry: RegistryCatalogEntry,
) -> Callable[
    [PseudoLabelAcceptancePolicyFactory],
    PseudoLabelAcceptancePolicyFactory,
]:
    """pseudo-label acceptance policy spec을 method hook registry에 등록한다."""

    def _decorator(
        factory: PseudoLabelAcceptancePolicyFactory,
    ) -> PseudoLabelAcceptancePolicyFactory:
        for policy_name in policy_names:
            normalized_name = policy_name.strip().lower()
            _PSEUDO_LABEL_ACCEPTANCE_POLICY_REGISTRY[normalized_name] = factory
            _PSEUDO_LABEL_ACCEPTANCE_POLICY_CATALOG[normalized_name] = catalog_entry
        return factory

    return _decorator


def build_pseudo_label_acceptance_policy(
    policy_name: str,
) -> PseudoLabelAcceptancePolicySpec:
    """acceptance policy 이름을 methods-owned selection policy spec으로 해석한다."""

    _import_ssl_hook_modules()
    normalized_name = policy_name.strip().lower()
    factory = _PSEUDO_LABEL_ACCEPTANCE_POLICY_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(f"Unsupported pseudo-label acceptance policy: {policy_name}.")


def list_pseudo_label_acceptance_policy_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 pseudo-label acceptance policy catalog entry 목록."""

    _import_ssl_hook_modules()
    return dedupe_registry_catalog_entries(
        tuple(_PSEUDO_LABEL_ACCEPTANCE_POLICY_CATALOG.values())
    )


def _import_ssl_hook_modules() -> None:
    package = importlib.import_module(_SSL_HOOKS_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.iter_modules(package_paths):
        if module_info.name in _SKIPPED_SSL_HOOK_MODULES:
            continue
        importlib.import_module(f"{_SSL_HOOKS_PACKAGE}.{module_info.name}")
