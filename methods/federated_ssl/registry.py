"""FL SSL method descriptor registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterable
from types import ModuleType

from methods.common.registry import MethodRegistry
from methods.federated_ssl.base import FederatedSslMethodDescriptor

_FEDERATED_SSL_PACKAGE = "methods.federated_ssl"
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
    """built-in FL SSL method module을 convention으로 import한다."""

    global _BUILTIN_FEDERATED_SSL_METHODS_LOADED
    if _BUILTIN_FEDERATED_SSL_METHODS_LOADED:
        return

    _import_federated_ssl_method_modules()
    _BUILTIN_FEDERATED_SSL_METHODS_LOADED = True


def resolve_federated_ssl_method_descriptor(
    name: str,
) -> FederatedSslMethodDescriptor:
    """method 이름을 FL SSL descriptor로 해석한다."""

    if _import_federated_ssl_method_module(name) is None:
        load_builtin_federated_ssl_methods()
    descriptor = _FEDERATED_SSL_METHOD_DESCRIPTORS.resolve(name)
    if descriptor is None:
        raise NotImplementedError(
            "Federated SSL method descriptor is not wired yet. "
            f"Choose one of {list(_FEDERATED_SSL_METHOD_DESCRIPTORS.names)} or add "
            f"the descriptor first: {name}"
        )
    return descriptor


def resolve_federated_ssl_method_descriptor_module(name: str) -> ModuleType:
    """method descriptor를 선언한 built-in module을 반환한다."""

    module = _import_federated_ssl_method_module(name)
    if module is not None:
        return module

    load_builtin_federated_ssl_methods()
    normalized_name = name.strip().lower().replace("-", "_")
    for module in _iter_federated_ssl_method_modules():
        if _module_declares_descriptor_name(module, normalized_name):
            return module

    raise ModuleNotFoundError(
        f"Federated SSL method descriptor module is not wired: {name}"
    )


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


def _import_federated_ssl_method_module(method_name: str) -> ModuleType | None:
    normalized_name = method_name.strip().lower().replace("-", "_")
    method_package = f"{_FEDERATED_SSL_PACKAGE}.{normalized_name}"
    module_name = f"{method_package}.descriptor"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {method_package, module_name}:
            raise
        return None
    _register_descriptor_from_module(module)
    return module


def _register_descriptor_from_module(module: ModuleType) -> None:
    descriptors = _module_descriptors(module)
    for descriptor in descriptors:
        _FEDERATED_SSL_METHOD_DESCRIPTORS.register(descriptor.name, item=descriptor)


def _import_federated_ssl_method_modules() -> None:
    for module in _iter_federated_ssl_method_modules():
        _register_descriptor_from_module(module)


def _iter_federated_ssl_method_modules() -> tuple[ModuleType, ...]:
    package = importlib.import_module(_FEDERATED_SSL_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return ()

    modules: list[ModuleType] = []
    for module_info in pkgutil.iter_modules(package_paths):
        if not module_info.ispkg:
            continue
        module = _import_federated_ssl_method_module(module_info.name)
        if module is not None:
            modules.append(module)
    return tuple(modules)


def _module_descriptors(
    module: ModuleType,
) -> tuple[FederatedSslMethodDescriptor, ...]:
    raw_descriptors = getattr(module, "descriptors", None)
    if raw_descriptors is None:
        descriptor = getattr(module, "descriptor", None)
        raw_descriptors = () if descriptor is None else (descriptor,)

    descriptors: list[FederatedSslMethodDescriptor] = []
    for descriptor in raw_descriptors:
        if not isinstance(descriptor, FederatedSslMethodDescriptor):
            raise TypeError(
                f"{module.__name__}.descriptors must contain "
                "FederatedSslMethodDescriptor."
            )
        descriptors.append(descriptor)
    return tuple(descriptors)


def _module_declares_descriptor_name(module: ModuleType, method_name: str) -> bool:
    return any(
        descriptor.name == method_name for descriptor in _module_descriptors(module)
    )
