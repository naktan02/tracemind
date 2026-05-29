"""FL SSL method name을 implementation family module로 해석한다."""

from __future__ import annotations

from importlib import import_module


def normalize_federated_ssl_method_name(method_name: str) -> str:
    """Hydra/CLI method 이름을 Python module 경로용 이름으로 정규화한다."""

    normalized = method_name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("method_name must not be empty.")
    return normalized


def resolve_federated_ssl_method_family_name(method_name: str) -> str:
    """variant method가 선언한 implementation family 이름을 읽는다."""

    normalized_name = normalize_federated_ssl_method_name(method_name)
    descriptor_module_name = f"methods.federated_ssl.{normalized_name}.descriptor"
    try:
        descriptor_module = import_module(descriptor_module_name)
    except ModuleNotFoundError as exc:
        if exc.name == descriptor_module_name:
            return normalized_name
        raise

    family_name = getattr(descriptor_module, "METHOD_IMPLEMENTATION_FAMILY", None)
    if family_name is None:
        return normalized_name
    return normalize_federated_ssl_method_name(str(family_name))
