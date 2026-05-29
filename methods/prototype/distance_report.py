"""Prototype centroid distance report core."""

from __future__ import annotations

import math
from dataclasses import dataclass

from methods.prototype.projections import (
    project_category_centroids_by_largest_cluster,
    require_single_category_centroids,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload

PROTOTYPE_CENTROID_VIEW_STRICT_SINGLE = "strict_single"
PROTOTYPE_CENTROID_VIEW_LARGEST_CLUSTER = "largest_cluster"
SUPPORTED_PROTOTYPE_CENTROID_VIEWS = (
    PROTOTYPE_CENTROID_VIEW_STRICT_SINGLE,
    PROTOTYPE_CENTROID_VIEW_LARGEST_CLUSTER,
)


@dataclass(frozen=True, slots=True)
class PrototypePairwiseDistanceReport:
    """Prototype centroid pairwise distance report payload."""

    categories: list[str]
    cosine_values: dict[tuple[str, str], float]
    l2_values: dict[tuple[str, str], float]


def resolve_prototype_centroid_view(
    *,
    payload: PrototypePackPayload,
    centroid_view: str,
) -> dict[str, list[float]]:
    """centroid_view 이름에 맞는 category centroid mapping을 만든다."""

    normalized_view = str(centroid_view).strip()
    if normalized_view == PROTOTYPE_CENTROID_VIEW_STRICT_SINGLE:
        return require_single_category_centroids(payload)
    if normalized_view == PROTOTYPE_CENTROID_VIEW_LARGEST_CLUSTER:
        return project_category_centroids_by_largest_cluster(payload)
    raise ValueError(
        "Unsupported prototype centroid_view. "
        f"Expected one of {SUPPORTED_PROTOTYPE_CENTROID_VIEWS}; "
        f"got {centroid_view!r}."
    )


def build_pairwise_distance_report(
    centroids: dict[str, list[float]],
) -> PrototypePairwiseDistanceReport:
    """category centroid 간 cosine similarity와 L2 distance를 계산한다."""

    categories = sorted(centroids)
    cosine_values: dict[tuple[str, str], float] = {}
    l2_values: dict[tuple[str, str], float] = {}
    for row_category in categories:
        for column_category in categories:
            left = centroids[row_category]
            right = centroids[column_category]
            cosine_values[(row_category, column_category)] = cosine_similarity(
                left,
                right,
            )
            l2_values[(row_category, column_category)] = l2_distance(left, right)
    return PrototypePairwiseDistanceReport(
        categories=categories,
        cosine_values=cosine_values,
        l2_values=l2_values,
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """두 vector의 cosine similarity를 계산한다."""

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm)


def l2_distance(left: list[float], right: list[float]) -> float:
    """두 vector의 L2 distance를 계산한다."""

    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


def render_pairwise_table(
    *,
    title: str,
    categories: list[str],
    values: dict[tuple[str, str], float],
) -> str:
    """pairwise metric table을 Markdown으로 렌더링한다."""

    header = ["category"] + categories
    lines = [
        title,
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row_category in categories:
        row = [row_category]
        for column_category in categories:
            row.append(f"{values[(row_category, column_category)]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
