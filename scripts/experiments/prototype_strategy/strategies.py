"""Prototype 생성 전략 모듈."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from methods.prototype.building.build_strategies import (
    DbscanPrototypeBuildStrategy as RuntimeDbscanPrototypeBuildStrategy,
)
from methods.prototype.building.build_strategies import (
    KMeansPrototypeBuildStrategy as RuntimeKMeansPrototypeBuildStrategy,
)
from methods.prototype.building.build_strategies import (
    PrototypeBuildRequest,
)
from methods.prototype.building.build_strategies import (
    SinglePrototypeBuildStrategy as RuntimeSinglePrototypeBuildStrategy,
)
from scripts.experiments.prototype_strategy.models import (
    PrototypeBuildStrategy,
    PrototypeIndex,
    PrototypeVector,
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
class SinglePrototypeStrategy:
    """카테고리당 단일 centroid를 만든다."""

    name: str = "single"

    def build(
        self,
        embeddings_by_label: Mapping[str, np.ndarray],
    ) -> PrototypeIndex:
        build_artifacts = RuntimeSinglePrototypeBuildStrategy(name=self.name).build(
            _build_runtime_request(embeddings_by_label)
        )
        return _prototype_index_from_pack(
            strategy_name=self.name,
            pack_payload=build_artifacts.pack_payload,
        )


@dataclass(slots=True)
class KMeansPrototypeStrategy:
    """카테고리별 k-means 기반 multi-prototype."""

    candidate_ks: tuple[int, ...] = (2, 3, 4, 5)
    silhouette_sample_size: int = 2000
    random_state: int = 42
    name: str = "kmeans"

    def build(
        self,
        embeddings_by_label: Mapping[str, np.ndarray],
    ) -> PrototypeIndex:
        build_artifacts = RuntimeKMeansPrototypeBuildStrategy(
            candidate_ks=self.candidate_ks,
            silhouette_sample_size=self.silhouette_sample_size,
            random_state=self.random_state,
            name=self.name,
        ).build(_build_runtime_request(embeddings_by_label))
        labels_metadata = {
            category: {
                "selected_k": len(prototypes),
                "fallback": len(prototypes) == 1,
                "member_counts": [
                    int(prototype.sample_count) for prototype in prototypes
                ],
            }
            for category, prototypes in sorted(
                build_artifacts.pack_payload.categories.items()
            )
        }
        return _prototype_index_from_pack(
            strategy_name=self.name,
            pack_payload=build_artifacts.pack_payload,
            metadata={
                "candidate_ks": list(self.candidate_ks),
                "silhouette_sample_size": self.silhouette_sample_size,
                "random_state": self.random_state,
                "labels": labels_metadata,
            },
        )


@dataclass(slots=True)
class DbscanPrototypeStrategy:
    """카테고리별 DBSCAN 기반 multi-prototype."""

    eps_values: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25)
    min_samples_values: tuple[int, ...] = (3, 5, 8)
    search_sample_size: int = 3000
    min_cluster_coverage: float = 0.6
    random_state: int = 42
    name: str = "dbscan"

    def build(
        self,
        embeddings_by_label: Mapping[str, np.ndarray],
    ) -> PrototypeIndex:
        build_artifacts = RuntimeDbscanPrototypeBuildStrategy(
            eps_values=self.eps_values,
            min_samples_values=self.min_samples_values,
            search_sample_size=self.search_sample_size,
            min_cluster_coverage=self.min_cluster_coverage,
            random_state=self.random_state,
            name=self.name,
        ).build(_build_runtime_request(embeddings_by_label))
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
    normalized_name = strategy_name.lower()
    if normalized_name == "single":
        return SinglePrototypeStrategy()
    if normalized_name == "kmeans":
        return KMeansPrototypeStrategy(
            candidate_ks=kmeans_candidate_ks,
            silhouette_sample_size=kmeans_silhouette_sample_size,
            random_state=seed,
        )
    if normalized_name == "dbscan":
        return DbscanPrototypeStrategy(
            eps_values=dbscan_eps_values,
            min_samples_values=dbscan_min_samples_values,
            search_sample_size=dbscan_search_sample_size,
            min_cluster_coverage=dbscan_min_cluster_coverage,
            random_state=seed,
        )
    raise ValueError(f"Unsupported strategy: {strategy_name}")


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
    normalized_name = strategy_name.lower()
    if normalized_name == "all":
        return (
            SinglePrototypeStrategy(),
            KMeansPrototypeStrategy(
                candidate_ks=kmeans_candidate_ks,
                silhouette_sample_size=kmeans_silhouette_sample_size,
                random_state=seed,
            ),
            DbscanPrototypeStrategy(
                eps_values=dbscan_eps_values,
                min_samples_values=dbscan_min_samples_values,
                search_sample_size=dbscan_search_sample_size,
                min_cluster_coverage=dbscan_min_cluster_coverage,
                random_state=seed,
            ),
        )
    return (
        build_requested_strategy(
            strategy_name=normalized_name,
            seed=seed,
            kmeans_candidate_ks=kmeans_candidate_ks,
            kmeans_silhouette_sample_size=kmeans_silhouette_sample_size,
            dbscan_eps_values=dbscan_eps_values,
            dbscan_min_samples_values=dbscan_min_samples_values,
            dbscan_search_sample_size=dbscan_search_sample_size,
            dbscan_min_cluster_coverage=dbscan_min_cluster_coverage,
        ),
    )


# 이전 실험 스크립트/노트북 호환용 alias
build_strategy = build_requested_strategy
build_strategies = build_requested_strategies
