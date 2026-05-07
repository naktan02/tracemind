"""SSL hook registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from methods.ssl.hooks.selection import PseudoLabelSelectionHook

PseudoLabelSelectionHookFactory = Callable[[], "PseudoLabelSelectionHook"]
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


def _import_ssl_hook_modules() -> None:
    package = importlib.import_module(_SSL_HOOKS_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.iter_modules(package_paths):
        if module_info.name in _SKIPPED_SSL_HOOK_MODULES:
            continue
        importlib.import_module(f"{_SSL_HOOKS_PACKAGE}.{module_info.name}")
