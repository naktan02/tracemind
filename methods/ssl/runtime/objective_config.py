"""Query SSL objective runtime config 해석."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from methods.adaptation.query_text_views.data import DEFAULT_STRONG_VIEW_POLICY
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


@dataclass(frozen=True, slots=True)
class QuerySslObjectiveRuntimeConfig:
    """Training objective의 `query_ssl.*` 축을 local trainer 입력으로 정규화한다."""

    method_name: str
    algorithm_name: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    strong_view_policy: str = DEFAULT_STRONG_VIEW_POLICY
    unlabeled_batch_size: int | None = None

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig | None,
    ) -> "QuerySslObjectiveRuntimeConfig | None":
        """Training objective extras에서 `query_ssl.*` 축을 읽는다."""

        if objective_config is None:
            return None
        extras = objective_config.get_component_extras("query_ssl")
        method_name = _optional_str(extras.get("method_name"))
        algorithm_name = _optional_str(extras.get("algorithm_name"))
        if method_name is None and algorithm_name is None:
            return None
        if method_name is None or algorithm_name is None:
            raise ValueError(
                "query_ssl objective extras require both method_name and "
                "algorithm_name."
            )
        return cls._from_parts(
            method_name=method_name,
            algorithm_name=algorithm_name,
            source=extras,
            strong_view_policy=(
                _optional_str(extras.get("strong_view_policy"))
                or DEFAULT_STRONG_VIEW_POLICY
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
        *,
        strong_view_policy: str = DEFAULT_STRONG_VIEW_POLICY,
    ) -> "QuerySslObjectiveRuntimeConfig":
        """Hydra `query_ssl_method` 같은 method mapping을 typed config로 해석한다."""

        method_name = _optional_str(source.get("name"))
        algorithm_name = _optional_str(source.get("algorithm_name"))
        if method_name is None or algorithm_name is None:
            raise ValueError("query_ssl_method requires name and algorithm_name.")
        return cls._from_parts(
            method_name=method_name,
            algorithm_name=algorithm_name,
            source=source,
            strong_view_policy=strong_view_policy,
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        method_name: str,
        algorithm_name: str,
        source: Mapping[str, object],
        strong_view_policy: str,
    ) -> "QuerySslObjectiveRuntimeConfig":
        parameters = {
            str(key): value
            for key, value in source.items()
            if str(key) not in _NON_PARAMETER_KEYS
        }
        unlabeled_batch_size = parameters.get("unlabeled_batch_size")
        normalized_unlabeled_batch_size = (
            None if unlabeled_batch_size is None else int(unlabeled_batch_size)
        )
        if (
            normalized_unlabeled_batch_size is not None
            and normalized_unlabeled_batch_size <= 0
        ):
            raise ValueError("query_ssl.unlabeled_batch_size must be positive.")
        return cls(
            method_name=method_name,
            algorithm_name=algorithm_name,
            parameters=parameters,
            strong_view_policy=strong_view_policy,
            unlabeled_batch_size=normalized_unlabeled_batch_size,
        )


_NON_PARAMETER_KEYS = frozenset(
    {
        "name",
        "method_name",
        "algorithm_name",
        "strong_view_policy",
    }
)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
