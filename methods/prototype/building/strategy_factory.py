"""Prototype build strategy factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from methods.prototype.building.base import PrototypeBuildStrategy
from methods.prototype.building.dbscan import DbscanPrototypeBuildStrategy
from methods.prototype.building.kmeans import KMeansPrototypeBuildStrategy
from methods.prototype.building.single import SinglePrototypeBuildStrategy


@dataclass(frozen=True, slots=True)
class PrototypeBuildStrategyConfig:
    """Shared prototype build strategy selection parameters."""

    seed: int
    kmeans_candidate_ks: tuple[int, ...]
    kmeans_silhouette_sample_size: int
    dbscan_eps_values: tuple[float, ...]
    dbscan_min_samples_values: tuple[int, ...]
    dbscan_search_sample_size: int
    dbscan_min_cluster_coverage: float


PrototypeBuildStrategyBuilder = Callable[
    [PrototypeBuildStrategyConfig],
    PrototypeBuildStrategy,
]


def build_prototype_build_strategy(
    *,
    strategy_name: str,
    config: PrototypeBuildStrategyConfig,
) -> PrototypeBuildStrategy:
    """Build one prototype strategy by canonical methods-owned name."""

    normalized_name = strategy_name.strip().lower()
    builders = _strategy_builders()
    builder = builders.get(normalized_name)
    if builder is None:
        raise ValueError(f"Unsupported prototype build strategy: {strategy_name}")
    return builder(config)


def build_prototype_build_strategies(
    *,
    strategy_name: str,
    config: PrototypeBuildStrategyConfig,
) -> tuple[PrototypeBuildStrategy, ...]:
    """Build one or all configured prototype strategies."""

    normalized_name = strategy_name.strip().lower()
    if normalized_name == "all":
        return tuple(builder(config) for builder in _strategy_builders().values())
    return (
        build_prototype_build_strategy(
            strategy_name=normalized_name,
            config=config,
        ),
    )


def _strategy_builders() -> dict[str, PrototypeBuildStrategyBuilder]:
    return {
        "single": _build_single_strategy,
        "kmeans": _build_kmeans_strategy,
        "dbscan": _build_dbscan_strategy,
    }


def _build_single_strategy(
    _config: PrototypeBuildStrategyConfig,
) -> PrototypeBuildStrategy:
    return SinglePrototypeBuildStrategy()


def _build_kmeans_strategy(
    config: PrototypeBuildStrategyConfig,
) -> PrototypeBuildStrategy:
    return KMeansPrototypeBuildStrategy(
        candidate_ks=config.kmeans_candidate_ks,
        silhouette_sample_size=config.kmeans_silhouette_sample_size,
        random_state=config.seed,
    )


def _build_dbscan_strategy(
    config: PrototypeBuildStrategyConfig,
) -> PrototypeBuildStrategy:
    return DbscanPrototypeBuildStrategy(
        eps_values=config.dbscan_eps_values,
        min_samples_values=config.dbscan_min_samples_values,
        search_sample_size=config.dbscan_search_sample_size,
        min_cluster_coverage=config.dbscan_min_cluster_coverage,
        random_state=config.seed,
    )
