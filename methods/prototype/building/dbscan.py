"""DBSCAN multi-prototype build strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
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

        categories_to_build = resolve_categories_to_build(request)
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
                        "centroid": mean_centroid(embeddings),
                        "sample_count": count,
                    }
                ]
                labels_metadata[category] = {
                    "fallback": True,
                    "reason": "too_few_examples",
                }
                continue

            sample_index = sample_indices(
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
                        "centroid": mean_centroid(embeddings),
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
                        "centroid": mean_centroid(cluster_members),
                        "sample_count": member_count,
                    }
                )

            noise_members = embeddings[final_labels < 0]
            if noise_members.size > 0:
                noise_count = int(noise_members.shape[0])
                prototypes.append(
                    {
                        "prototype_id": f"{category}:dbscan:noise",
                        "centroid": mean_centroid(noise_members),
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
