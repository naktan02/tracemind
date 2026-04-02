"""Prototype 생성 전략 모듈."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from scripts.experiments.prototype_strategy.models import (
    PrototypeBuildStrategy,
    PrototypeIndex,
    PrototypeVector,
)
from scripts.experiments.prototype_strategy.sampling import sample_index_array
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.services.prototypes.build_strategies import (
    KMeansPrototypeBuildStrategy as RuntimeKMeansPrototypeBuildStrategy,
)
from shared.src.services.prototypes.build_strategies import (
    PrototypeBuildRequest,
)
from shared.src.services.prototypes.build_strategies import (
    SinglePrototypeBuildStrategy as RuntimeSinglePrototypeBuildStrategy,
)

_EXPERIMENT_BUILT_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


def normalize_vector(values: Sequence[float]) -> list[float]:
    """벡터를 L2 정규화한다."""
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        raise ValueError("Prototype centroid must not have zero norm.")
    return (array / norm).tolist()


def mean_centroid(embeddings: np.ndarray) -> list[float]:
    """임베딩 평균 centroid를 정규화해 반환한다."""
    return normalize_vector(embeddings.mean(axis=0))


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
        categories: dict[str, list[PrototypeVector]] = {}
        metadata: dict[str, dict[str, float | int | bool | list[int] | str]] = {}

        for label, embeddings in sorted(embeddings_by_label.items()):
            count = int(embeddings.shape[0])
            if count < 2:
                categories[label] = [
                    PrototypeVector(
                        prototype_id=f"{label}:single",
                        centroid=mean_centroid(embeddings),
                        member_count=count,
                    )
                ]
                metadata[label] = {
                    "fallback": True,
                    "reason": "too_few_examples",
                }
                continue

            sample_index = sample_index_array(
                count,
                limit=self.search_sample_size,
                seed=self.random_state,
            )
            sample_embeddings = embeddings[sample_index]

            best_params: tuple[float, int] | None = None
            best_score = -1.0
            best_coverage = 0.0

            for eps in self.eps_values:
                for min_samples in self.min_samples_values:
                    labels = DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                        metric="cosine",
                    ).fit_predict(sample_embeddings)
                    unique_labels = sorted(
                        int(cluster_id)
                        for cluster_id in np.unique(labels)
                        if cluster_id >= 0
                    )
                    if len(unique_labels) < 2:
                        continue

                    covered_mask = labels >= 0
                    coverage = float(np.mean(covered_mask))
                    if coverage < self.min_cluster_coverage:
                        continue

                    filtered_embeddings = sample_embeddings[covered_mask]
                    filtered_labels = labels[covered_mask]
                    unique_filtered = np.unique(filtered_labels)
                    if unique_filtered.shape[0] < 2:
                        continue

                    silhouette = float(
                        silhouette_score(
                            filtered_embeddings,
                            filtered_labels,
                            metric="cosine",
                        )
                    )
                    score = silhouette * coverage
                    if score > best_score:
                        best_score = score
                        best_coverage = coverage
                        best_params = (eps, min_samples)

            if best_params is None:
                categories[label] = [
                    PrototypeVector(
                        prototype_id=f"{label}:single",
                        centroid=mean_centroid(embeddings),
                        member_count=count,
                    )
                ]
                metadata[label] = {
                    "fallback": True,
                    "reason": "no_valid_cluster",
                }
                continue

            best_eps, best_min_samples = best_params
            final_labels = DBSCAN(
                eps=best_eps,
                min_samples=best_min_samples,
                metric="cosine",
            ).fit_predict(embeddings)

            cluster_ids = sorted(
                int(cluster_id)
                for cluster_id in np.unique(final_labels)
                if cluster_id >= 0
            )
            prototypes: list[PrototypeVector] = []
            member_counts: list[int] = []
            for cluster_id in cluster_ids:
                cluster_members = embeddings[final_labels == cluster_id]
                member_count = int(cluster_members.shape[0])
                member_counts.append(member_count)
                prototypes.append(
                    PrototypeVector(
                        prototype_id=f"{label}:dbscan:{cluster_id}",
                        centroid=mean_centroid(cluster_members),
                        member_count=member_count,
                    )
                )

            noise_members = embeddings[final_labels < 0]
            if noise_members.size > 0:
                prototypes.append(
                    PrototypeVector(
                        prototype_id=f"{label}:dbscan:noise",
                        centroid=mean_centroid(noise_members),
                        member_count=int(noise_members.shape[0]),
                    )
                )
                member_counts.append(int(noise_members.shape[0]))

            categories[label] = prototypes
            metadata[label] = {
                "selected_eps": best_eps,
                "selected_min_samples": best_min_samples,
                "silhouette": best_score / best_coverage,
                "coverage": best_coverage,
                "fallback": False,
                "member_counts": member_counts,
            }

        return PrototypeIndex(
            strategy_name=self.name,
            categories=categories,
            metadata={
                "eps_values": list(self.eps_values),
                "min_samples_values": list(self.min_samples_values),
                "labels": metadata,
            },
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
