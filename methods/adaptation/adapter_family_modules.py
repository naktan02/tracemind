"""Adapter family implementation module path resolution."""

from __future__ import annotations

_ADAPTATION_PACKAGE = "methods.adaptation"
_ADAPTER_FAMILY_MODULE_ROOTS = {
    "peft_classifier": "methods.adaptation.peft_text_classifier",
}


def adapter_family_module_name(
    *,
    adapter_kind: str,
    submodule: str,
) -> str:
    """adapter kind가 소유한 implementation submodule 경로를 반환한다."""

    normalized_adapter_kind = normalize_adapter_kind(adapter_kind)
    normalized_submodule = submodule.strip(".")
    if not normalized_submodule:
        raise ValueError("submodule must not be empty.")
    module_root = _ADAPTER_FAMILY_MODULE_ROOTS.get(
        normalized_adapter_kind,
        f"{_ADAPTATION_PACKAGE}.{normalized_adapter_kind.replace('-', '_')}",
    )
    return f"{module_root}.{normalized_submodule}"


def normalize_adapter_kind(adapter_kind: str) -> str:
    """dispatcher registry key로 쓸 adapter kind를 정규화한다."""

    normalized_adapter_kind = adapter_kind.strip().lower()
    if not normalized_adapter_kind:
        raise ValueError("adapter_kind must not be empty.")
    return normalized_adapter_kind
