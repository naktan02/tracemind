"""SSL hook registry."""

from __future__ import annotations

from collections.abc import Callable

from methods.ssl.hooks.selection import (
    FixedConfidencePseudoLabelSelectionHook,
    MarginThresholdPseudoLabelSelectionHook,
    PseudoLabelSelectionHook,
)

PseudoLabelSelectionHookFactory = Callable[[], PseudoLabelSelectionHook]

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

    normalized_name = hook_name.strip().lower()
    factory = _PSEUDO_LABEL_SELECTION_HOOK_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(f"Unsupported pseudo-label selection hook: {hook_name}.")


register_pseudo_label_selection_hook("top1_confidence_only")(
    FixedConfidencePseudoLabelSelectionHook
)
register_pseudo_label_selection_hook("top1_margin_threshold")(
    MarginThresholdPseudoLabelSelectionHook
)
