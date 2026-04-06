"""PrototypePackPayload projection helpers."""

from __future__ import annotations

from shared.src.contracts.prototype_contracts import PrototypePackPayload


def require_single_category_centroids(
    payload: PrototypePackPayload,
) -> dict[str, list[float]]:
    """모든 category가 prototype 하나일 때만 centroid view를 반환한다."""
    centroids: dict[str, list[float]] = {}
    multi_categories: list[str] = []
    for category, prototypes in payload.categories.items():
        if len(prototypes) != 1:
            multi_categories.append(category)
            continue
        centroids[category] = list(prototypes[0].centroid)
    if multi_categories:
        raise ValueError(
            "require_single_category_centroids only supports "
            f"single-prototype categories: {sorted(multi_categories)}"
        )
    return centroids


def project_category_centroids_by_largest_cluster(
    payload: PrototypePackPayload,
) -> dict[str, list[float]]:
    """category별 sample_count가 가장 큰 prototype을 대표 centroid로 고른다."""
    centroids: dict[str, list[float]] = {}
    for category, prototypes in payload.categories.items():
        if not prototypes:
            raise ValueError(f"Category '{category}' does not contain any prototypes.")
        representative = max(
            prototypes,
            key=lambda prototype: (
                int(prototype.sample_count),
                "" if prototype.prototype_id is None else str(prototype.prototype_id),
            ),
        )
        centroids[category] = list(representative.centroid)
    return centroids


__all__ = [
    "project_category_centroids_by_largest_cluster",
    "require_single_category_centroids",
]
