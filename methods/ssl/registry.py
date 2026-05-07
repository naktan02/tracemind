"""Query SSL algorithm registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from methods.common.registry import MethodRegistry

from .base import QuerySslAlgorithm

QuerySslAlgorithmFactory = Callable[[Mapping[str, Any]], QuerySslAlgorithm]

_QUERY_SSL_ALGORITHM_REGISTRY = MethodRegistry[QuerySslAlgorithmFactory](
    item_label="query SSL algorithm",
)
_BUILTIN_QUERY_SSL_ALGORITHMS_LOADED = False


def register_query_ssl_algorithm(
    *algorithm_names: str,
) -> Callable[[QuerySslAlgorithmFactory], QuerySslAlgorithmFactory]:
    """algorithm_name으로 Query SSL algorithm factory를 등록하는 decorator."""

    def _decorator(factory: QuerySslAlgorithmFactory) -> QuerySslAlgorithmFactory:
        return _QUERY_SSL_ALGORITHM_REGISTRY.register(
            *algorithm_names,
            item=factory,
        )

    return _decorator


def load_builtin_query_ssl_algorithms() -> None:
    """built-in Query SSL algorithm module을 명시적으로 import한다."""

    global _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED
    if _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED:
        return

    from .fixmatch import fixmatch as _fixmatch_algorithm  # noqa: F401

    _BUILTIN_QUERY_SSL_ALGORITHMS_LOADED = True


def build_query_ssl_algorithm(
    *,
    algorithm_name: str,
    parameters: Mapping[str, Any],
) -> QuerySslAlgorithm:
    """algorithm_name과 parameter로 Query SSL algorithm을 생성한다."""

    load_builtin_query_ssl_algorithms()
    factory = _QUERY_SSL_ALGORITHM_REGISTRY.resolve(algorithm_name)
    if factory is None:
        raise ValueError(f"Unsupported query SSL algorithm: {algorithm_name}.")
    return factory(parameters)
