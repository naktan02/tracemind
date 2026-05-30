"""FL SSL method name을 implementation family module로 해석한다."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_METHOD_SURFACE_LEAF_ALIASES = frozenset(
    {
        "round_policy",
        "runtime_requirements",
        "server_policy",
        "server_step_parameters",
    }
)


def normalize_federated_ssl_method_name(method_name: str) -> str:
    """Hydra/CLI method 이름을 Python module 경로용 이름으로 정규화한다."""

    normalized = method_name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("method_name must not be empty.")
    return normalized


def resolve_federated_ssl_method_family_name(method_name: str) -> str:
    """variant method가 선언한 implementation family 이름을 읽는다."""

    normalized_name = normalize_federated_ssl_method_name(method_name)
    method_package = f"methods.federated_ssl.{normalized_name}"
    descriptor_module_name = f"methods.federated_ssl.{normalized_name}.descriptor"
    try:
        descriptor_module = import_module(descriptor_module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {method_package, descriptor_module_name}:
            descriptor_module = None
        else:
            raise
    family_name = (
        None
        if descriptor_module is None
        else getattr(descriptor_module, "METHOD_IMPLEMENTATION_FAMILY", None)
    )
    if family_name is None:
        try:
            from methods.federated_ssl.registry import (
                resolve_federated_ssl_method_descriptor_module,
            )

            descriptor_module = resolve_federated_ssl_method_descriptor_module(
                normalized_name
            )
        except ModuleNotFoundError:
            return normalized_name
        family_name = _method_surface_value(
            descriptor_module,
            normalized_name,
            "METHOD_IMPLEMENTATION_FAMILY",
            None,
        )
        if family_name is None:
            return normalized_name
    return normalize_federated_ssl_method_name(str(family_name))


def import_method_family_module(
    *,
    method_name: str,
    module_leaf: str,
) -> ModuleType | None:
    """method implementation family 아래 optional module을 convention으로 import한다."""

    family_name = resolve_federated_ssl_method_family_name(method_name)
    normalized_leaf = module_leaf.strip().removesuffix(".py")
    if not normalized_leaf:
        raise ValueError("module_leaf must not be empty.")
    module_name = f"methods.federated_ssl.{family_name}.{normalized_leaf}"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            if normalized_leaf in _METHOD_SURFACE_LEAF_ALIASES:
                return _import_optional_method_surface_module(family_name)
            return None
        raise


def _import_optional_method_surface_module(family_name: str) -> ModuleType | None:
    module_name = f"methods.federated_ssl.{family_name}.method_surface"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return None
        raise


def _method_surface_value(
    descriptor_module: object,
    method_name: str,
    attribute_name: str,
    default: object,
) -> object:
    surface_by_name = getattr(
        descriptor_module,
        "METHOD_CONFIG_SURFACE_BY_METHOD_NAME",
        {},
    )
    if isinstance(surface_by_name, dict):
        method_surface = surface_by_name.get(method_name, {})
        if isinstance(method_surface, dict) and attribute_name in method_surface:
            return method_surface[attribute_name]
    return getattr(descriptor_module, attribute_name, default)
