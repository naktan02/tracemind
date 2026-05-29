"""Prototype builder가 공유하는 벡터 유틸리티."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from methods.prototype.building.base import PrototypeBuildRequest


def normalize_vector(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("Prototype centroid must not have zero norm.")
    return (array / norm).tolist()


def mean_centroid(embeddings: np.ndarray) -> list[float]:
    return normalize_vector(embeddings.mean(axis=0))


def sample_indices(
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


def resolve_categories_to_build(
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
