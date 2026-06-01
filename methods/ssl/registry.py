"""Query SSL algorithm registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from typing import Any

from methods.common.registry import MethodRegistry

from .base import (
    QuerySslAlgorithm,
    QuerySslAlgorithmDescriptor,
    QuerySslAlgorithmFactory,
    QuerySslRequiredViews,
    QuerySslRuntimeRequirements,
)

_QUERY_SSL_ALGORITHM_REGISTRY = MethodRegistry[QuerySslAlgorithmDescriptor](
    item_label="query SSL algorithm descriptor",
)
_BUILTIN_QUERY_SSL_ALGORITHMS_LOADED = False
_QUERY_SSL_ALGORITHMS_PACKAGE = "methods.ssl.algorithms"
_SKIPPED_QUERY_SSL_ALGORITHM_MODULES = frozenset(
    {
        "base",
        "registry",
    }
)


def register_query_ssl_algorithm(
    *algorithm_names: str,
    required_views: QuerySslRequiredViews,
    display_name: str | None = None,
    default_uses_labeled_batches: bool = True,
    runtime_requirements: QuerySslRuntimeRequirements | None = None,
) -> Callable[[QuerySslAlgorithmFactory], QuerySslAlgorithmFactory]:
    """algorithm_name으로 Query SSL algorithm factory를 등록하는 decorator."""

    def _decorator(factory: QuerySslAlgorithmFactory) -> QuerySslAlgorithmFactory:
        names = tuple(algorithm_names)
        if not names:
            raise ValueError("query SSL algorithm registry names must not be empty.")
        primary_name = names[0]
        descriptor = QuerySslAlgorithmDescriptor(
            algorithm_name=primary_name,
            display_name=display_name or primary_name,
            required_views=required_views,
            algorithm_factory=factory,
            default_uses_labeled_batches=default_uses_labeled_batches,
            runtime_requirements=(
                runtime_requirements or QuerySslRuntimeRequirements()
            ),
        )
        _QUERY_SSL_ALGORITHM_REGISTRY.register(*names, item=descriptor)
        return factory

    return _decorator


def load_builtin_query_ssl_algorithms() -> None:
    """built-in Query SSL algorithm module을 convention으로 import한다."""

    global _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED
    if _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED:
        return

    package = importlib.import_module(_QUERY_SSL_ALGORITHMS_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is not None:
        for module_info in pkgutil.iter_modules(package_paths):
            if module_info.name in _SKIPPED_QUERY_SSL_ALGORITHM_MODULES:
                continue
            if not module_info.ispkg:
                continue
            importlib.import_module(
                f"{_QUERY_SSL_ALGORITHMS_PACKAGE}.{module_info.name}.{module_info.name}"
            )

    _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED = True


def build_query_ssl_algorithm(
    *,
    algorithm_name: str,
    parameters: Mapping[str, Any],
) -> QuerySslAlgorithm:
    """algorithm_name과 parameter로 Query SSL algorithm을 생성한다."""

    descriptor = resolve_query_ssl_algorithm_descriptor(algorithm_name)
    return descriptor.build_algorithm(parameters)


def resolve_query_ssl_algorithm_descriptor(
    algorithm_name: str,
) -> QuerySslAlgorithmDescriptor:
    """algorithm_name을 Query SSL method descriptor로 해석한다."""

    load_builtin_query_ssl_algorithms()
    descriptor = _QUERY_SSL_ALGORITHM_REGISTRY.resolve(algorithm_name)
    if descriptor is None:
        raise ValueError(f"Unsupported query SSL algorithm: {algorithm_name}.")
    return descriptor
