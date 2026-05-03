"""Query SSL algorithm registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import QuerySslAlgorithm

QuerySslAlgorithmFactory = Callable[[Mapping[str, Any]], QuerySslAlgorithm]

_QUERY_SSL_ALGORITHM_REGISTRY: dict[str, QuerySslAlgorithmFactory] = {}


def register_query_ssl_algorithm(
    *algorithm_names: str,
) -> Callable[[QuerySslAlgorithmFactory], QuerySslAlgorithmFactory]:
    """algorithm_name으로 Query SSL algorithm factory를 등록하는 decorator."""

    def _decorator(factory: QuerySslAlgorithmFactory) -> QuerySslAlgorithmFactory:
        for algorithm_name in algorithm_names:
            _QUERY_SSL_ALGORITHM_REGISTRY[algorithm_name.strip().lower()] = factory
        return factory

    return _decorator


def build_query_ssl_algorithm(
    *,
    algorithm_name: str,
    parameters: Mapping[str, Any],
) -> QuerySslAlgorithm:
    """algorithm_name과 parameter로 Query SSL algorithm을 생성한다."""

    normalized_name = algorithm_name.strip().lower()
    factory = _QUERY_SSL_ALGORITHM_REGISTRY.get(normalized_name)
    if factory is None:
        raise ValueError(f"Unsupported query SSL algorithm: {algorithm_name}.")
    return factory(parameters)


# Built-in algorithms self-register via decorators when imported.
from .fixmatch import algorithm as _fixmatch_algorithm  # noqa: E402,F401
