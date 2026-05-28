"""Payload adapter implementation module path resolution."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

_ADAPTATION_PACKAGE = "methods.adaptation"
_IMPLEMENTATION_ROOT_PACKAGES = (
    _ADAPTATION_PACKAGE,
    "methods.classification",
)
_PAYLOAD_MODULE_MANIFEST = "payload_adapter_module"
_PAYLOAD_ADAPTER_MODULE_ROOTS: dict[str, str] = {}


def payload_adapter_module_root(payload_adapter_kind: str) -> str:
    """payload adapter kind가 소유한 implementation module root를 반환한다."""

    normalized_payload_adapter_kind = normalize_payload_adapter_kind(
        payload_adapter_kind
    )
    module_root = _PAYLOAD_ADAPTER_MODULE_ROOTS.get(normalized_payload_adapter_kind)
    if module_root is not None:
        return module_root
    _import_conventional_payload_manifest(normalized_payload_adapter_kind)
    module_root = _PAYLOAD_ADAPTER_MODULE_ROOTS.get(normalized_payload_adapter_kind)
    if module_root is not None:
        return module_root
    _import_all_payload_manifests()
    return _PAYLOAD_ADAPTER_MODULE_ROOTS.get(
        normalized_payload_adapter_kind,
        _default_adaptation_module_root(normalized_payload_adapter_kind),
    )


def register_payload_adapter_module_root(
    *payload_adapter_kinds: str,
    module_root: str,
) -> None:
    """구현 owner 옆 manifest에서 payload adapter kind별 module root를 등록한다."""

    normalized_module_root = module_root.strip()
    if not normalized_module_root:
        raise ValueError("module_root must not be empty.")
    for payload_adapter_kind in _normalized_payload_adapter_kinds(
        payload_adapter_kinds
    ):
        registered_root = _PAYLOAD_ADAPTER_MODULE_ROOTS.get(payload_adapter_kind)
        if registered_root is not None and registered_root != normalized_module_root:
            raise ValueError(
                "Duplicate payload adapter module root registration: "
                f"{payload_adapter_kind} -> {registered_root} / "
                f"{normalized_module_root}"
            )
        _PAYLOAD_ADAPTER_MODULE_ROOTS[payload_adapter_kind] = normalized_module_root


def payload_adapter_module_name(
    *,
    payload_adapter_kind: str,
    submodule: str,
) -> str:
    """payload adapter kind가 소유한 implementation submodule 경로를 반환한다."""

    normalized_payload_adapter_kind = normalize_payload_adapter_kind(
        payload_adapter_kind
    )
    normalized_submodule = submodule.strip(".")
    if not normalized_submodule:
        raise ValueError("submodule must not be empty.")
    return (
        f"{payload_adapter_module_root(normalized_payload_adapter_kind)}."
        f"{normalized_submodule}"
    )


def normalize_payload_adapter_kind(payload_adapter_kind: str) -> str:
    """dispatcher registry key로 쓸 payload adapter kind를 정규화한다."""

    normalized_payload_adapter_kind = payload_adapter_kind.strip().lower()
    if not normalized_payload_adapter_kind:
        raise ValueError("payload_adapter_kind must not be empty.")
    return normalized_payload_adapter_kind


def _normalized_payload_adapter_kinds(
    payload_adapter_kinds: Iterable[str],
) -> tuple[str, ...]:
    normalized = tuple(
        normalize_payload_adapter_kind(kind) for kind in payload_adapter_kinds
    )
    if not normalized:
        raise ValueError("at least one payload_adapter_kind must be provided.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"payload_adapter_kinds must be unique: {normalized!r}.")
    return normalized


def _import_conventional_payload_manifest(
    normalized_payload_adapter_kind: str,
) -> None:
    package_name = normalized_payload_adapter_kind.replace("-", "_")
    for root_package in _IMPLEMENTATION_ROOT_PACKAGES:
        _try_import_module(
            f"{root_package}.{package_name}.{_PAYLOAD_MODULE_MANIFEST}"
        )


def _import_all_payload_manifests() -> None:
    for root_package_name in _IMPLEMENTATION_ROOT_PACKAGES:
        root_package = importlib.import_module(root_package_name)
        package_paths = getattr(root_package, "__path__", None)
        if package_paths is None:
            continue
        for module_info in pkgutil.iter_modules(package_paths):
            if not module_info.ispkg:
                continue
            _try_import_module(
                f"{root_package_name}.{module_info.name}.{_PAYLOAD_MODULE_MANIFEST}"
            )


def _default_adaptation_module_root(normalized_payload_adapter_kind: str) -> str:
    return f"{_ADAPTATION_PACKAGE}.{normalized_payload_adapter_kind.replace('-', '_')}"


def _try_import_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return False
        raise
    return True
