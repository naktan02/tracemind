"""Query SSL adaptation objective registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import QuerySslObjective
from .fixmatch import FixMatchConfig, FixMatchObjective

QuerySslObjectiveFactory = Callable[[Mapping[str, Any]], QuerySslObjective]

_QUERY_SSL_OBJECTIVE_REGISTRY: dict[str, QuerySslObjectiveFactory] = {}


def register_query_ssl_objective(
    *objective_names: str,
    factory: QuerySslObjectiveFactory,
) -> None:
    """algorithm_name으로 Query SSL objective 구현을 등록한다."""

    for objective_name in objective_names:
        _QUERY_SSL_OBJECTIVE_REGISTRY[objective_name.strip().lower()] = factory


def build_query_ssl_objective(
    *,
    objective_name: str,
    parameters: Mapping[str, Any],
) -> QuerySslObjective:
    """algorithm_name과 method parameter로 objective adapter를 생성한다."""

    normalized_name = objective_name.strip().lower()
    factory = _QUERY_SSL_OBJECTIVE_REGISTRY.get(normalized_name)
    if factory is None:
        raise ValueError(
            f"Unsupported query SSL adaptation objective: {objective_name}."
        )
    return factory(parameters)


def _build_fixmatch_objective(parameters: Mapping[str, Any]) -> FixMatchObjective:
    return FixMatchObjective(
        config=FixMatchConfig(
            temperature=float(parameters["temperature"]),
            p_cutoff=float(parameters["p_cutoff"]),
            hard_label=bool(parameters.get("hard_label", True)),
            lambda_u=float(parameters.get("lambda_u", 1.0)),
            supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
        )
    )


register_query_ssl_objective("fixmatch", factory=_build_fixmatch_objective)


__all__ = [
    "QuerySslObjectiveFactory",
    "build_query_ssl_objective",
    "register_query_ssl_objective",
]
