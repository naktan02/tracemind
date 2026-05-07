"""FL SSL method descriptor registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from methods.common.registry import MethodRegistry
from methods.federated_ssl.base import FederatedSslMethodDescriptor

_FEDERATED_SSL_METHOD_DESCRIPTORS = MethodRegistry[FederatedSslMethodDescriptor](
    item_label="federated SSL method descriptor",
)
_BUILTIN_FEDERATED_SSL_METHODS_LOADED = False


def register_federated_ssl_method_descriptor(
    *method_names: str,
    descriptor: FederatedSslMethodDescriptor | None = None,
) -> (
    Callable[[FederatedSslMethodDescriptor], FederatedSslMethodDescriptor]
    | FederatedSslMethodDescriptor
):
    """method 이름으로 FL SSL method descriptor를 등록한다."""

    def _decorator(
        item: FederatedSslMethodDescriptor,
    ) -> FederatedSslMethodDescriptor:
        names = tuple(method_names) or (item.name,)
        return _FEDERATED_SSL_METHOD_DESCRIPTORS.register(*names, item=item)

    if descriptor is not None:
        return _decorator(descriptor)
    return _decorator


def load_builtin_federated_ssl_methods() -> None:
    """built-in FL SSL method module을 명시적으로 import한다."""

    global _BUILTIN_FEDERATED_SSL_METHODS_LOADED
    if _BUILTIN_FEDERATED_SSL_METHODS_LOADED:
        return

    from methods.federated_ssl.fedavg_pseudo_label import (  # noqa: F401
        fedavg_pseudo_label as _fedavg_pseudo_label,
    )

    _BUILTIN_FEDERATED_SSL_METHODS_LOADED = True


def resolve_federated_ssl_method_descriptor(
    name: str,
) -> FederatedSslMethodDescriptor:
    """method 이름을 FL SSL descriptor로 해석한다."""

    load_builtin_federated_ssl_methods()
    descriptor = _FEDERATED_SSL_METHOD_DESCRIPTORS.resolve(name)
    if descriptor is None:
        raise NotImplementedError(
            "Federated SSL method descriptor is not wired yet. "
            f"Choose one of {list(_FEDERATED_SSL_METHOD_DESCRIPTORS.names)} or add "
            f"the descriptor first: {name}"
        )
    return descriptor


def list_federated_ssl_method_descriptors(
    *,
    method_names: Iterable[str] | None = None,
) -> tuple[FederatedSslMethodDescriptor, ...]:
    """등록된 FL SSL method descriptor를 중복 없이 반환한다."""

    load_builtin_federated_ssl_methods()
    if method_names is not None:
        descriptors = {
            resolve_federated_ssl_method_descriptor(method_name)
            for method_name in method_names
        }
        return tuple(sorted(descriptors, key=lambda descriptor: descriptor.name))

    return _FEDERATED_SSL_METHOD_DESCRIPTORS.list_unique(
        names=None,
        sort_key=lambda descriptor: descriptor.name,
    )
