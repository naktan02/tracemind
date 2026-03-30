"""Production prototype 생성 전략."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scripts.prototypes.prototype_pack_builder import PrototypePackBuilder
from shared.src.contracts.prototype_build_state_contracts import (
    PrototypeBuildStatePayload,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload


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
    build_state_payload: PrototypeBuildStatePayload | None = None


class PrototypeBuildStrategy(Protocol):
    """Production prototype 생성 전략 공통 인터페이스."""

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
            pack_payload=PrototypePackPayload.model_validate(
                {
                    "schema_version": pack.schema_version,
                    "prototype_version": pack.prototype_version,
                    "embedding_model_id": pack.embedding_model_id,
                    "embedding_model_revision": pack.embedding_model_revision,
                    "translation_model_id": pack.translation_model_id,
                    "translation_model_revision": pack.translation_model_revision,
                    "translation_direction": pack.translation_direction,
                    "mapping_version": pack.mapping_version,
                    "build_method": pack.build_method,
                    "distance_metric": pack.distance_metric,
                    "built_at": pack.built_at,
                    "categories": {
                        category: [
                            {
                                "prototype_id": f"{category}:single",
                                "centroid": prototype.centroid,
                                "sample_count": prototype.sample_count,
                            }
                        ]
                        for category, prototype in pack.categories.items()
                    },
                }
            ),
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
            pack_payload=PrototypePackPayload.model_validate(
                {
                    "schema_version": "prototype_pack.v1",
                    "prototype_version": request.prototype_version,
                    "embedding_model_id": request.embedding_model_id,
                    "embedding_model_revision": request.embedding_model_revision,
                    "translation_model_id": request.translation_model_id,
                    "translation_model_revision": request.translation_model_revision,
                    "translation_direction": request.translation_direction,
                    "mapping_version": request.mapping_version,
                    "build_method": "kmeans_mean_centroid_l2_normalized",
                    "distance_metric": "cosine",
                    "built_at": request.built_at,
                    "categories": categories,
                }
            )
        )
