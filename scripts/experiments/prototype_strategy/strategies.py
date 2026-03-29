"""Prototype 생성 전략 모듈."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from scripts.experiments.prototype_strategy.models import (
    PrototypeIndex,
    PrototypeVector,
)


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


def sample_indices(
    count: int,
    *,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    """count 중 limit개 인덱스를 재현 가능하게 샘플링한다."""
    indices = np.arange(count)
    if limit is None or count <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


class MultiPrototypeScorer:
    """category별 여러 prototype 중 최대 cosine score를 선택한다."""

    def score(
        self,
        embedding: Sequence[float],
        prototype_index: PrototypeIndex,
    ) -> dict[str, float]:
        vector = np.asarray(embedding, dtype=np.float64)
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0.0:
            raise ValueError("Embedding norm must not be zero.")
        normalized = vector / vector_norm

        scores: dict[str, float] = {}
        for category, prototypes in prototype_index.categories.items():
            best_score = max(
                float(np.dot(normalized, np.asarray(prototype.centroid)))
                for prototype in prototypes
            )
            scores[category] = best_score
        return scores


@dataclass(slots=True)
class SinglePrototypeStrategy:
    """카테고리당 단일 centroid를 만든다."""

    name: str = "single"

    def build(
        self,
        embeddings_by_label: Mapping[str, np.ndarray],
    ) -> PrototypeIndex:
        categories: dict[str, list[PrototypeVector]] = {}
        for label, embeddings in sorted(embeddings_by_label.items()):
            categories[label] = [
                PrototypeVector(
                    prototype_id=f"{label}:single",
                    centroid=mean_centroid(embeddings),
                    member_count=int(embeddings.shape[0]),
                )
            ]
        return PrototypeIndex(
            strategy_name=self.name,
            categories=categories,
            metadata={"build_method": "single_mean_centroid"},
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
        categories: dict[str, list[PrototypeVector]] = {}
        metadata: dict[str, dict[str, float | int | bool | list[int]]] = {}

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
                    "selected_k": 1,
                    "silhouette": 0.0,
                    "fallback": True,
                }
                continue

            sample_index = sample_indices(
                count,
                limit=self.silhouette_sample_size,
                seed=self.random_state,
            )
            sample_embeddings = embeddings[sample_index]

            best_k = 1
            best_silhouette = -1.0
            for candidate_k in self.candidate_ks:
                if candidate_k >= count:
                    continue
                labels = KMeans(
                    n_clusters=candidate_k,
                    random_state=self.random_state,
                    n_init="auto",
                ).fit_predict(sample_embeddings)
                unique_labels = np.unique(labels)
                if unique_labels.shape[0] < 2:
                    continue
                silhouette = float(silhouette_score(sample_embeddings, labels))
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = candidate_k

            if best_k == 1:
                categories[label] = [
                    PrototypeVector(
                        prototype_id=f"{label}:single",
                        centroid=mean_centroid(embeddings),
                        member_count=count,
                    )
                ]
                metadata[label] = {
                    "selected_k": 1,
                    "silhouette": 0.0,
                    "fallback": True,
                }
                continue

            fitted = KMeans(
                n_clusters=best_k,
                random_state=self.random_state,
                n_init="auto",
            ).fit(embeddings)

            prototypes: list[PrototypeVector] = []
            member_counts: list[int] = []
            for cluster_id in range(best_k):
                cluster_members = embeddings[fitted.labels_ == cluster_id]
                member_count = int(cluster_members.shape[0])
                member_counts.append(member_count)
                prototypes.append(
                    PrototypeVector(
                        prototype_id=f"{label}:kmeans:{cluster_id}",
                        centroid=mean_centroid(cluster_members),
                        member_count=member_count,
                    )
                )
            categories[label] = prototypes
            metadata[label] = {
                "selected_k": best_k,
                "silhouette": best_silhouette,
                "fallback": False,
                "member_counts": member_counts,
            }

        return PrototypeIndex(
            strategy_name=self.name,
            categories=categories,
            metadata={
                "candidate_ks": list(self.candidate_ks),
                "labels": metadata,
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

            sample_index = sample_indices(
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
