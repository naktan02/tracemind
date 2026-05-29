"""Prototype 생성 전략 모듈."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from methods.prototype.building.base import (
    PrototypeBuildRequest,
)
from methods.prototype.building.base import (
    PrototypeBuildStrategy as RuntimePrototypeBuildStrategy,
)
from methods.prototype.building.strategy_factory import (
    PrototypeBuildStrategyConfig,
    build_prototype_build_strategies,
    build_prototype_build_strategy,
)
from methods.prototype.index import PrototypeIndex, PrototypeVector
from scripts.experiments.prototype_analysis.prototype_strategy.models import (
    PrototypeBuildStrategy,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload

_EXPERIMENT_BUILT_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _build_runtime_request(
    embeddings_by_label: Mapping[str, np.ndarray],
) -> PrototypeBuildRequest:
    return PrototypeBuildRequest(
        embeddings_by_category={
            label: np.asarray(embeddings, dtype=np.float64).tolist()
            for label, embeddings in sorted(embeddings_by_label.items())
        },
        prototype_version="experiment_runtime_preview",
        embedding_backend="experiment",
        embedding_model_id="experiment",
        embedding_model_revision="preview",
        mapping_version="experiment_runtime_preview",
        built_at=_EXPERIMENT_BUILT_AT,
    )


def _prototype_index_from_pack(
    *,
    strategy_name: str,
    pack_payload: PrototypePackPayload,
    metadata: Mapping[str, object] | None = None,
) -> PrototypeIndex:
    categories = {
        category: [
            PrototypeVector(
                prototype_id=prototype.prototype_id or f"{category}:{index}",
                centroid=list(prototype.centroid),
                member_count=int(prototype.sample_count),
            )
            for index, prototype in enumerate(prototypes)
        ]
        for category, prototypes in sorted(pack_payload.categories.items())
    }
    payload_metadata: dict[str, object] = {
        "build_method": pack_payload.build_method,
        "distance_metric": pack_payload.distance_metric,
        "mapping_version": pack_payload.mapping_version,
        "source": "shared_runtime",
    }
    if metadata is not None:
        payload_metadata.update(dict(metadata))
    return PrototypeIndex(
        strategy_name=strategy_name,
        categories=categories,
        metadata=payload_metadata,
    )


@dataclass(slots=True)
class ExperimentPrototypeBuildStrategy:
    """methods-owned prototype build strategy를 실험용 index로 변환하는 adapter."""

    runtime_strategy: RuntimePrototypeBuildStrategy

    @property
    def name(self) -> str:
        return self.runtime_strategy.name

    def build(
        self,
        embeddings_by_label: Mapping[str, np.ndarray],
    ) -> PrototypeIndex:
        build_artifacts = self.runtime_strategy.build(
            _build_runtime_request(embeddings_by_label)
        )
        return _prototype_index_from_pack(
            strategy_name=self.name,
            pack_payload=build_artifacts.pack_payload,
            metadata=build_artifacts.metadata,
        )


def build_requested_strategy(
    *,
    strategy_name: str,
    seed: int,
    kmeans_candidate_ks: tuple[int, ...],
    kmeans_silhouette_sample_size: int,
    dbscan_eps_values: tuple[float, ...],
    dbscan_min_samples_values: tuple[int, ...],
    dbscan_search_sample_size: int,
    dbscan_min_cluster_coverage: float,
) -> PrototypeBuildStrategy:
    return ExperimentPrototypeBuildStrategy(
        runtime_strategy=build_prototype_build_strategy(
            strategy_name=strategy_name,
            config=_build_strategy_config(
                seed=seed,
                kmeans_candidate_ks=kmeans_candidate_ks,
                kmeans_silhouette_sample_size=kmeans_silhouette_sample_size,
                dbscan_eps_values=dbscan_eps_values,
                dbscan_min_samples_values=dbscan_min_samples_values,
                dbscan_search_sample_size=dbscan_search_sample_size,
                dbscan_min_cluster_coverage=dbscan_min_cluster_coverage,
            ),
        )
    )


def build_requested_strategies(
    *,
    strategy_name: str,
    seed: int,
    kmeans_candidate_ks: tuple[int, ...],
    kmeans_silhouette_sample_size: int,
    dbscan_eps_values: tuple[float, ...],
    dbscan_min_samples_values: tuple[int, ...],
    dbscan_search_sample_size: int,
    dbscan_min_cluster_coverage: float,
) -> tuple[PrototypeBuildStrategy, ...]:
    return tuple(
        ExperimentPrototypeBuildStrategy(runtime_strategy=runtime_strategy)
        for runtime_strategy in build_prototype_build_strategies(
            strategy_name=strategy_name,
            config=_build_strategy_config(
                seed=seed,
                kmeans_candidate_ks=kmeans_candidate_ks,
                kmeans_silhouette_sample_size=kmeans_silhouette_sample_size,
                dbscan_eps_values=dbscan_eps_values,
                dbscan_min_samples_values=dbscan_min_samples_values,
                dbscan_search_sample_size=dbscan_search_sample_size,
                dbscan_min_cluster_coverage=dbscan_min_cluster_coverage,
            ),
        )
    )


def _build_strategy_config(
    *,
    seed: int,
    kmeans_candidate_ks: tuple[int, ...],
    kmeans_silhouette_sample_size: int,
    dbscan_eps_values: tuple[float, ...],
    dbscan_min_samples_values: tuple[int, ...],
    dbscan_search_sample_size: int,
    dbscan_min_cluster_coverage: float,
) -> PrototypeBuildStrategyConfig:
    return PrototypeBuildStrategyConfig(
        seed=seed,
        kmeans_candidate_ks=kmeans_candidate_ks,
        kmeans_silhouette_sample_size=kmeans_silhouette_sample_size,
        dbscan_eps_values=dbscan_eps_values,
        dbscan_min_samples_values=dbscan_min_samples_values,
        dbscan_search_sample_size=dbscan_search_sample_size,
        dbscan_min_cluster_coverage=dbscan_min_cluster_coverage,
    )
