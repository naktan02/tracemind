"""Query SSL algorithm registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from methods.common.registry import MethodRegistry

from .base import (
    QuerySslAlgorithm,
    QuerySslAlgorithmDescriptor,
    QuerySslAlgorithmFactory,
    QuerySslRequiredViews,
)

_QUERY_SSL_ALGORITHM_REGISTRY = MethodRegistry[QuerySslAlgorithmDescriptor](
    item_label="query SSL algorithm descriptor",
)
_BUILTIN_QUERY_SSL_ALGORITHMS_LOADED = False


def register_query_ssl_algorithm(
    *algorithm_names: str,
    required_views: QuerySslRequiredViews,
    display_name: str | None = None,
    default_uses_labeled_batches: bool = True,
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
        )
        _QUERY_SSL_ALGORITHM_REGISTRY.register(*names, item=descriptor)
        return factory

    return _decorator


def load_builtin_query_ssl_algorithms() -> None:
    """built-in Query SSL algorithm module을 명시적으로 import한다."""

    global _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED
    if _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED:
        return

    from .algorithms.fixmatch import fixmatch as _fixmatch_algorithm  # noqa: F401

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
