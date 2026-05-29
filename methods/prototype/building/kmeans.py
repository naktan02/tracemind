"""K-means multi-prototype build strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from methods.prototype.building.base import (
    PrototypeBuildArtifacts,
    PrototypeBuildRequest,
)
from methods.prototype.building.vector_ops import (
    mean_centroid,
    resolve_categories_to_build,
    sample_indices,
)
from shared.src.services.prototypes.payload_serialization import (
    PrototypePackPayloadSpec,
    build_prototype_pack_payload,
)


@dataclass(slots=True)
class KMeansPrototypeBuildStrategy:
    """카테고리별 k-means 기반 multi-prototype 생성 전략."""

    candidate_ks: Sequence[int] = (2, 3, 4, 5)
    silhouette_sample_size: int = 2000
    random_state: int = 42
    name: str = "kmeans"
    supports_exact_build_state: bool = False

    def build(self, request: PrototypeBuildRequest) -> PrototypeBuildArtifacts:
        if request.built_at is None:
            raise ValueError("built_at must not be None.")

        categories_to_build = resolve_categories_to_build(request)
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
                        "centroid": mean_centroid(embeddings),
                        "sample_count": count,
                    }
                ]
                continue

            sample_index = sample_indices(
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
                        "centroid": mean_centroid(embeddings),
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
                        "centroid": mean_centroid(cluster_members),
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
