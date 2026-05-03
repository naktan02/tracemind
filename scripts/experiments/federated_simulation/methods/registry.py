"""FL SSL method runtime registry."""

from __future__ import annotations

from collections.abc import Callable

from .base import FederatedSslMethodDescriptor, FederatedSslMethodRuntime

FederatedSslMethodFactory = Callable[[], FederatedSslMethodRuntime]

_REGISTERED_METHODS: dict[
    str,
    tuple[FederatedSslMethodDescriptor, FederatedSslMethodFactory],
] = {}


def register_federated_ssl_method(
    *method_names: str,
    descriptor: FederatedSslMethodDescriptor,
) -> Callable[[FederatedSslMethodFactory], FederatedSslMethodFactory]:
    """method 이름으로 FL SSL runtime factory를 등록한다."""

    def _decorator(
        factory: FederatedSslMethodFactory,
    ) -> FederatedSslMethodFactory:
        registered_method = (descriptor, factory)
        for method_name in method_names:
            _REGISTERED_METHODS[method_name.strip().lower()] = registered_method
        return factory

    return _decorator


def resolve_federated_ssl_method(name: str) -> FederatedSslMethodDescriptor:
    """method 이름을 baseline descriptor로 해석한다."""

    normalized_name = name.strip().lower()
    registered_method = _REGISTERED_METHODS.get(normalized_name)
    if registered_method is None:
        raise NotImplementedError(
            "Federated SSL method is not wired yet. "
            f"Choose one of {sorted(_REGISTERED_METHODS)} or add the method "
            f"descriptor first: {name}"
        )
    descriptor, _factory = registered_method
    return descriptor


def build_federated_ssl_method_runtime(name: str) -> FederatedSslMethodRuntime:
    """method 이름으로 simulation runtime 조합을 생성한다."""

    normalized_name = name.strip().lower()
    registered_method = _REGISTERED_METHODS.get(normalized_name)
    if registered_method is None:
        raise NotImplementedError(
            "Federated SSL method is not wired yet. "
            f"Choose one of {sorted(_REGISTERED_METHODS)} or add the method "
            f"runtime first: {name}"
        )
    _descriptor, factory = registered_method
    return factory()


# Built-in methods self-register via decorators when imported.
from . import fedavg_pseudo_label as _fedavg_pseudo_label  # noqa: E402,F401
