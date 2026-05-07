"""SSL hook registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from methods.ssl.hooks.selection import PseudoLabelSelectionHook

PseudoLabelSelectionHookFactory = Callable[[], "PseudoLabelSelectionHook"]

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

    _ensure_builtin_ssl_hooks_loaded()
    normalized_name = hook_name.strip().lower()
    factory = _PSEUDO_LABEL_SELECTION_HOOK_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(f"Unsupported pseudo-label selection hook: {hook_name}.")


def _ensure_builtin_ssl_hooks_loaded() -> None:
    from methods.ssl.hooks.builtin_loader import load_builtin_ssl_hooks

    load_builtin_ssl_hooks()
