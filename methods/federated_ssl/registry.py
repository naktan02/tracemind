"""FL SSL method descriptor registry."""

from __future__ import annotations

from collections.abc import Iterable

from methods.federated_ssl.base import FederatedSslMethodDescriptor

_FEDERATED_SSL_METHOD_DESCRIPTORS: dict[str, FederatedSslMethodDescriptor] = {}


def register_federated_ssl_method_descriptor(
    *method_names: str,
    descriptor: FederatedSslMethodDescriptor,
) -> FederatedSslMethodDescriptor:
    """method 이름으로 FL SSL method descriptor를 등록한다."""

    names = tuple(method_names) or (descriptor.name,)
    for method_name in names:
        _FEDERATED_SSL_METHOD_DESCRIPTORS[method_name.strip().lower()] = descriptor
    return descriptor


def resolve_federated_ssl_method_descriptor(
    name: str,
) -> FederatedSslMethodDescriptor:
    """method 이름을 FL SSL descriptor로 해석한다."""

    normalized_name = name.strip().lower()
    descriptor = _FEDERATED_SSL_METHOD_DESCRIPTORS.get(normalized_name)
    if descriptor is None:
        raise NotImplementedError(
            "Federated SSL method descriptor is not wired yet. "
            f"Choose one of {sorted(_FEDERATED_SSL_METHOD_DESCRIPTORS)} or add "
            f"the descriptor first: {name}"
        )
    return descriptor


def list_federated_ssl_method_descriptors(
    *,
    method_names: Iterable[str] | None = None,
) -> tuple[FederatedSslMethodDescriptor, ...]:
    """등록된 FL SSL method descriptor를 중복 없이 반환한다."""

    if method_names is None:
        descriptors = set(_FEDERATED_SSL_METHOD_DESCRIPTORS.values())
    else:
        descriptors = {
            resolve_federated_ssl_method_descriptor(method_name)
            for method_name in method_names
        }
    return tuple(sorted(descriptors, key=lambda descriptor: descriptor.name))


# Built-in descriptors self-register via decorators when imported.
from methods.federated_ssl.fedavg_pseudo_label import (  # noqa: E402,F401
    fedavg_pseudo_label as _fedavg_pseudo_label,
)
