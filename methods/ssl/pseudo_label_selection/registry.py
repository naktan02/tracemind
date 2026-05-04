"""Pseudo-label selection method registry."""

from __future__ import annotations

from collections.abc import Callable

from methods.ssl.pseudo_label_selection.base import PseudoLabelSelectionMethod

PseudoLabelSelectionMethodFactory = Callable[[], PseudoLabelSelectionMethod]

_PSEUDO_LABEL_SELECTION_METHOD_REGISTRY: dict[
    str, PseudoLabelSelectionMethodFactory
] = {}


def register_pseudo_label_selection_method(
    *method_names: str,
) -> Callable[[PseudoLabelSelectionMethodFactory], PseudoLabelSelectionMethodFactory]:
    """이름으로 pseudo-label selection method를 등록하는 decorator."""

    def _decorator(
        factory: PseudoLabelSelectionMethodFactory,
    ) -> PseudoLabelSelectionMethodFactory:
        for method_name in method_names:
            _PSEUDO_LABEL_SELECTION_METHOD_REGISTRY[method_name.strip().lower()] = (
                factory
            )
        return factory

    return _decorator


def build_pseudo_label_selection_method(
    method_name: str,
) -> PseudoLabelSelectionMethod:
    """method 이름으로 selection method 구현을 생성한다."""

    normalized_name = method_name.strip().lower()
    factory = _PSEUDO_LABEL_SELECTION_METHOD_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(f"Unsupported pseudo-label selection method: {method_name}.")


def build_pseudo_label_selection_hook(
    hook_name: str,
) -> PseudoLabelSelectionMethod:
    """legacy hook 용어를 method registry로 해석한다."""

    return build_pseudo_label_selection_method(hook_name)


# Built-in selection methods self-register via decorators when imported.
from methods.ssl.pseudo_label_selection import (  # noqa: E402,F401
    fixed_confidence as _fixed_confidence,
)
from methods.ssl.pseudo_label_selection import (  # noqa: E402,F401
    margin_threshold as _margin_threshold,
)
