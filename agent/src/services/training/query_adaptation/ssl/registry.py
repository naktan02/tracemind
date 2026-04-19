"""Query adaptation SSL selection algorithm registry."""

from __future__ import annotations

from collections.abc import Callable

from .algorithms.fixed_confidence import FixedConfidenceQuerySslAlgorithm
from .algorithms.margin_threshold import MarginThresholdQuerySslAlgorithm
from .base import QuerySslAlgorithm

QuerySslAlgorithmFactory = Callable[[], QuerySslAlgorithm]

_QUERY_SSL_ALGORITHM_REGISTRY: dict[str, QuerySslAlgorithmFactory] = {}


def register_query_ssl_algorithm(
    *algorithm_names: str,
    factory: QuerySslAlgorithmFactory,
) -> None:
    """이름으로 query adaptation SSL algorithm 구현을 등록한다."""

    for algorithm_name in algorithm_names:
        _QUERY_SSL_ALGORITHM_REGISTRY[algorithm_name.strip().lower()] = factory


def build_query_ssl_algorithm(
    algorithm_name: str,
) -> QuerySslAlgorithm:
    """알고리즘 이름으로 selection algorithm 구현을 생성한다."""

    normalized_name = algorithm_name.strip().lower()
    factory = _QUERY_SSL_ALGORITHM_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory()
    raise ValueError(
        f"Unsupported query adaptation SSL algorithm: {algorithm_name}."
    )


register_query_ssl_algorithm(
    "top1_margin_threshold",
    factory=MarginThresholdQuerySslAlgorithm,
)
register_query_ssl_algorithm(
    "top1_confidence_only",
    factory=FixedConfidenceQuerySslAlgorithm,
)


__all__ = [
    "QuerySslAlgorithmFactory",
    "build_query_ssl_algorithm",
    "register_query_ssl_algorithm",
]
