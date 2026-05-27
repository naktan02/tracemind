"""Adapter family implementation module path resolution."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

_ADAPTATION_PACKAGE = "methods.adaptation"
_IMPLEMENTATION_ROOT_PACKAGES = (
    _ADAPTATION_PACKAGE,
    "methods.classification",
)
_FAMILY_MODULE_MANIFEST = "adapter_family_module"
_ADAPTER_FAMILY_MODULE_ROOTS: dict[str, str] = {}


def adapter_family_module_root(adapter_kind: str) -> str:
    """adapter kind가 소유한 implementation module root를 반환한다."""

    normalized_adapter_kind = normalize_adapter_kind(adapter_kind)
    module_root = _ADAPTER_FAMILY_MODULE_ROOTS.get(normalized_adapter_kind)
    if module_root is not None:
        return module_root
    _import_conventional_family_manifest(normalized_adapter_kind)
    module_root = _ADAPTER_FAMILY_MODULE_ROOTS.get(normalized_adapter_kind)
    if module_root is not None:
        return module_root
    _import_all_family_manifests()
    return _ADAPTER_FAMILY_MODULE_ROOTS.get(
        normalized_adapter_kind,
        _default_adaptation_module_root(normalized_adapter_kind),
    )


def register_adapter_family_module_root(
    *adapter_kinds: str,
    module_root: str,
) -> None:
    """family 구현 옆 manifest에서 adapter kind별 module root를 등록한다."""

    normalized_module_root = module_root.strip()
    if not normalized_module_root:
        raise ValueError("module_root must not be empty.")
    for adapter_kind in _normalized_adapter_kinds(adapter_kinds):
        registered_root = _ADAPTER_FAMILY_MODULE_ROOTS.get(adapter_kind)
        if registered_root is not None and registered_root != normalized_module_root:
            raise ValueError(
                "Duplicate adapter family module root registration: "
                f"{adapter_kind} -> {registered_root} / {normalized_module_root}"
            )
        _ADAPTER_FAMILY_MODULE_ROOTS[adapter_kind] = normalized_module_root


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
    return (
        f"{adapter_family_module_root(normalized_adapter_kind)}.{normalized_submodule}"
    )


def normalize_adapter_kind(adapter_kind: str) -> str:
    """dispatcher registry key로 쓸 adapter kind를 정규화한다."""

    normalized_adapter_kind = adapter_kind.strip().lower()
    if not normalized_adapter_kind:
        raise ValueError("adapter_kind must not be empty.")
    return normalized_adapter_kind


def _normalized_adapter_kinds(adapter_kinds: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_adapter_kind(kind) for kind in adapter_kinds)
    if not normalized:
        raise ValueError("at least one adapter_kind must be provided.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"adapter_kinds must be unique: {normalized!r}.")
    return normalized


def _import_conventional_family_manifest(normalized_adapter_kind: str) -> None:
    family_package = normalized_adapter_kind.replace("-", "_")
    for root_package in _IMPLEMENTATION_ROOT_PACKAGES:
        _try_import_module(f"{root_package}.{family_package}.{_FAMILY_MODULE_MANIFEST}")


def _import_all_family_manifests() -> None:
    for root_package_name in _IMPLEMENTATION_ROOT_PACKAGES:
        root_package = importlib.import_module(root_package_name)
        package_paths = getattr(root_package, "__path__", None)
        if package_paths is None:
            continue
        for module_info in pkgutil.iter_modules(package_paths):
            if not module_info.ispkg:
                continue
            _try_import_module(
                f"{root_package_name}.{module_info.name}.{_FAMILY_MODULE_MANIFEST}"
            )


def _default_adaptation_module_root(normalized_adapter_kind: str) -> str:
    return f"{_ADAPTATION_PACKAGE}.{normalized_adapter_kind.replace('-', '_')}"


def _try_import_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return False
        raise
    return True
