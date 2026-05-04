"""공용 prototype 생성 전략."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from shared.src.contracts.prototype_build_state_contracts import (
    SinglePrototypeBuildStatePayload,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.services.prototypes.payload_serialization import (
    PrototypePackPayloadSpec,
    build_prototype_pack_payload,
    build_single_prototype_pack_payload,
)
from shared.src.services.prototypes.prototype_pack_builder import PrototypePackBuilder


@dataclass(slots=True)
class PrototypeBuildRequest:
    """Prototype 생성에 필요한 공통 입력."""

    embeddings_by_category: Mapping[str, Sequence[Sequence[float]]]
    prototype_version: str
    embedding_backend: str
    embedding_model_id: str
    embedding_model_revision: str
    normalize_embeddings: bool = True
    task_prefix: str = ""
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    mapping_version: str = ""
    built_at: datetime | None = None
    required_categories: Sequence[str] | None = None


@dataclass(slots=True)
class PrototypeBuildArtifacts:
    """생성 전략 결과물."""

    pack_payload: PrototypePackPayload
    build_state_payload: SinglePrototypeBuildStatePayload | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PrototypeBuildStrategy(Protocol):
    """prototype 생성 전략 공통 인터페이스."""

    name: str
    supports_exact_build_state: bool

    def build(self, request: PrototypeBuildRequest) -> PrototypeBuildArtifacts:
        """입력 임베딩으로 pack/build-state 결과물을 생성한다."""


def describe_prototype_build_strategy(strategy: object) -> dict[str, Any]:
    """manifest에 남길 전략 설명을 만든다."""
    if is_dataclass(strategy):
        return asdict(strategy)
    return {
        "name": getattr(strategy, "name", type(strategy).__name__),
        "class_name": type(strategy).__name__,
    }


def _normalize_vector(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("Prototype centroid must not have zero norm.")
    return (array / norm).tolist()


def _mean_centroid(embeddings: np.ndarray) -> list[float]:
    return _normalize_vector(embeddings.mean(axis=0))


def _sample_indices(
    count: int,
    *,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    indices = np.arange(count)
    if limit is None or count <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def _resolve_categories_to_build(
    request: PrototypeBuildRequest,
) -> tuple[str, ...]:
    categories_to_build = (
        tuple(request.required_categories)
        if request.required_categories is not None
        else tuple(sorted(request.embeddings_by_category))
    )
    if not categories_to_build:
        raise ValueError("At least one category is required to build a prototype pack.")

    for category in categories_to_build:
        bucket = request.embeddings_by_category.get(category)
        if not bucket:
            raise ValueError(f"Category '{category}' has no embeddings to build from.")
    return categories_to_build


@dataclass(slots=True)
class SinglePrototypeBuildStrategy:
    """기존 exact mean-centroid builder를 감싼 single 전략."""

    name: str = "single"
    supports_exact_build_state: bool = True

    def build(self, request: PrototypeBuildRequest) -> PrototypeBuildArtifacts:
        if request.built_at is None:
            raise ValueError("built_at must not be None.")

        builder = PrototypePackBuilder()
        build_state = builder.build_state(
            request.embeddings_by_category,
            prototype_version=request.prototype_version,
            embedding_backend=request.embedding_backend,
            embedding_model_id=request.embedding_model_id,
            embedding_model_revision=request.embedding_model_revision,
            normalize_embeddings=request.normalize_embeddings,
            task_prefix=request.task_prefix,
            translation_model_id=request.translation_model_id,
            translation_model_revision=request.translation_model_revision,
            translation_direction=request.translation_direction,
            mapping_version=request.mapping_version,
            built_at=request.built_at,
            required_categories=request.required_categories,
        )
        pack = builder.build_pack_from_state(build_state)
        return PrototypeBuildArtifacts(
            pack_payload=build_single_prototype_pack_payload(pack),
            build_state_payload=build_state,
        )


@dataclass(slots=True)
class KMeansPrototypeBuildStrategy:
    """카테고리별 k-means multi-prototype 생성 전략."""

    candidate_ks: Sequence[int] = (2, 3, 4, 5)
    silhouette_sample_size: int = 2000
    random_state: int = 42
    name: str = "kmeans"
    supports_exact_build_state: bool = False

    def build(self, request: PrototypeBuildRequest) -> PrototypeBuildArtifacts:
        if request.built_at is None:
            raise ValueError("built_at must not be None.")

        categories_to_build = _resolve_categories_to_build(request)
        categories: dict[str, list[dict[str, object]]] = {}
        candidate_ks = tuple(int(value) for value in self.candidate_ks)

        for category in categories_to_build:
            bucket = request.embeddings_by_category[category]
            embeddings = np.asarray(bucket, dtype=np.float64)
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                raise ValueError(
                    f"Category '{category}' has invalid embeddings for k-means build."
                )
            count = int(embeddings.shape[0])

            if count < 2:
                categories[category] = [
                    {
                        "prototype_id": f"{category}:single",
                        "centroid": _mean_centroid(embeddings),
                        "sample_count": count,
                    }
                ]
                continue

            sample_index = _sample_indices(
                count,
                limit=self.silhouette_sample_size,
                seed=self.random_state,
            )
            sample_embeddings = embeddings[sample_index]

            best_k = 1
            best_silhouette = -1.0
            for candidate_k in candidate_ks:
                if candidate_k >= count:
                    continue
                labels = KMeans(
                    n_clusters=candidate_k,
                    random_state=self.random_state,
                    n_init="auto",
                ).fit_predict(sample_embeddings)
                if np.unique(labels).shape[0] < 2:
                    continue
                silhouette = float(silhouette_score(sample_embeddings, labels))
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = candidate_k

            if best_k == 1:
                categories[category] = [
                    {
                        "prototype_id": f"{category}:single",
                        "centroid": _mean_centroid(embeddings),
                        "sample_count": count,
                    }
                ]
                continue

            fitted = KMeans(
                n_clusters=best_k,
                random_state=self.random_state,
                n_init="auto",
            ).fit(embeddings)
            prototypes: list[dict[str, object]] = []
            for cluster_id in range(best_k):
                cluster_members = embeddings[fitted.labels_ == cluster_id]
                member_count = int(cluster_members.shape[0])
                prototypes.append(
                    {
                        "prototype_id": f"{category}:kmeans:{cluster_id}",
                        "centroid": _mean_centroid(cluster_members),
                        "sample_count": member_count,
                    }
                )
            categories[category] = prototypes

        return PrototypeBuildArtifacts(
            pack_payload=build_prototype_pack_payload(
                spec=PrototypePackPayloadSpec(
                    schema_version="prototype_pack.v1",
                    prototype_version=request.prototype_version,
                    embedding_model_id=request.embedding_model_id,
                    embedding_model_revision=request.embedding_model_revision,
                    translation_model_id=request.translation_model_id,
                    translation_model_revision=request.translation_model_revision,
                    translation_direction=request.translation_direction,
                    mapping_version=request.mapping_version,
                    build_method="kmeans_mean_centroid_l2_normalized",
                    distance_metric="cosine",
                    built_at=request.built_at,
                ),
                categories=categories,
            )
        )


@dataclass(slots=True)
class DbscanPrototypeBuildStrategy:
    """카테고리별 DBSCAN 기반 multi-prototype 생성 전략."""

    eps_values: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.25)
    min_samples_values: Sequence[int] = (3, 5, 8)
    search_sample_size: int = 3000
    min_cluster_coverage: float = 0.6
    random_state: int = 42
    name: str = "dbscan"
    supports_exact_build_state: bool = False

    def build(self, request: PrototypeBuildRequest) -> PrototypeBuildArtifacts:
        if request.built_at is None:
            raise ValueError("built_at must not be None.")

        categories_to_build = _resolve_categories_to_build(request)
        categories: dict[str, list[dict[str, object]]] = {}
        labels_metadata: dict[str, dict[str, object]] = {}
        eps_values = tuple(float(value) for value in self.eps_values)
        min_samples_values = tuple(int(value) for value in self.min_samples_values)

        for category in categories_to_build:
            bucket = request.embeddings_by_category[category]
            embeddings = np.asarray(bucket, dtype=np.float64)
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                raise ValueError(
                    f"Category '{category}' has invalid embeddings for DBSCAN build."
                )
            count = int(embeddings.shape[0])

            if count < 2:
                categories[category] = [
                    {
                        "prototype_id": f"{category}:single",
                        "centroid": _mean_centroid(embeddings),
                        "sample_count": count,
                    }
                ]
                labels_metadata[category] = {
                    "fallback": True,
                    "reason": "too_few_examples",
                }
                continue

            sample_index = _sample_indices(
                count,
                limit=self.search_sample_size,
                seed=self.random_state,
            )
            sample_embeddings = embeddings[sample_index]

            best_params: tuple[float, int] | None = None
            best_score = -1.0
            best_coverage = 0.0

            for eps in eps_values:
                for min_samples in min_samples_values:
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
                categories[category] = [
                    {
                        "prototype_id": f"{category}:single",
                        "centroid": _mean_centroid(embeddings),
                        "sample_count": count,
                    }
                ]
                labels_metadata[category] = {
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
            prototypes: list[dict[str, object]] = []
            member_counts: list[int] = []
            for cluster_id in cluster_ids:
                cluster_members = embeddings[final_labels == cluster_id]
                member_count = int(cluster_members.shape[0])
                member_counts.append(member_count)
                prototypes.append(
                    {
                        "prototype_id": f"{category}:dbscan:{cluster_id}",
                        "centroid": _mean_centroid(cluster_members),
                        "sample_count": member_count,
                    }
                )

            noise_members = embeddings[final_labels < 0]
            if noise_members.size > 0:
                noise_count = int(noise_members.shape[0])
                prototypes.append(
                    {
                        "prototype_id": f"{category}:dbscan:noise",
                        "centroid": _mean_centroid(noise_members),
                        "sample_count": noise_count,
                    }
                )
                member_counts.append(noise_count)

            categories[category] = prototypes
            labels_metadata[category] = {
                "selected_eps": best_eps,
                "selected_min_samples": best_min_samples,
                "silhouette": best_score / best_coverage,
                "coverage": best_coverage,
                "fallback": False,
                "member_counts": member_counts,
            }

        return PrototypeBuildArtifacts(
            pack_payload=build_prototype_pack_payload(
                spec=PrototypePackPayloadSpec(
                    schema_version="prototype_pack.v1",
                    prototype_version=request.prototype_version,
                    embedding_model_id=request.embedding_model_id,
                    embedding_model_revision=request.embedding_model_revision,
                    translation_model_id=request.translation_model_id,
                    translation_model_revision=request.translation_model_revision,
                    translation_direction=request.translation_direction,
                    mapping_version=request.mapping_version,
                    build_method="dbscan_mean_centroid_l2_normalized",
                    distance_metric="cosine",
                    built_at=request.built_at,
                ),
                categories=categories,
            ),
            metadata={
                "eps_values": list(eps_values),
                "min_samples_values": list(min_samples_values),
                "search_sample_size": self.search_sample_size,
                "min_cluster_coverage": self.min_cluster_coverage,
                "random_state": self.random_state,
                "labels": labels_metadata,
            },
        )
