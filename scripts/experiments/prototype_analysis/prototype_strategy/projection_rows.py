"""Projection row와 visual center 계산 helper."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from scripts.experiments.prototype_analysis.prototype_strategy.models import (
    PrototypeIndex,
    PrototypeVector,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

ProjectionRowValue = float | int | str | None
ProjectionPointRow = dict[str, float | str]
ProjectionOverlayRow = dict[str, ProjectionRowValue]


def build_projected_point_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    projection: np.ndarray,
) -> list[ProjectionPointRow]:
    """샘플 row와 2D projection 값을 JSONL row로 변환한다."""

    return [
        {
            "query_id": row.get("query_id", ""),
            "label": row["mapped_label_4"],
            "x": float(point[0]),
            "y": float(point[1]),
        }
        for row, point in zip(rows, projection, strict=True)
    ]


def project_prototypes(
    *,
    reducer: Any,
    prototype_index: PrototypeIndex | None,
) -> list[ProjectionOverlayRow]:
    """Prototype centroid를 fitted reducer의 2D 공간으로 투영한다."""

    if prototype_index is None:
        return []

    indexed_prototypes: list[tuple[str, PrototypeVector]] = []
    for label, prototypes in sorted(prototype_index.categories.items()):
        for prototype in prototypes:
            indexed_prototypes.append((label, prototype))

    if not indexed_prototypes:
        return []

    projected = reducer.transform(
        np.asarray(
            [prototype.centroid for _, prototype in indexed_prototypes],
            dtype=np.float64,
        )
    )
    return [
        {
            "label": label,
            "prototype_id": prototype.prototype_id,
            "member_count": prototype.member_count,
            "x": float(point[0]),
            "y": float(point[1]),
        }
        for (label, prototype), point in zip(
            indexed_prototypes,
            projected,
            strict=True,
        )
    ]


def build_visual_center_points(
    *,
    rows: Sequence[LabeledQueryRow],
    embeddings: np.ndarray,
    sample_index: np.ndarray,
    point_rows: Sequence[ProjectionPointRow],
    prototype_index: PrototypeIndex | None,
    seed: int,
) -> list[ProjectionOverlayRow]:
    """Label 또는 kmeans cluster별 projected visual center를 만든다."""

    if prototype_index is not None and prototype_index.strategy_name == "kmeans":
        return _build_kmeans_visual_centers(
            rows=rows,
            embeddings=embeddings,
            sample_index=sample_index,
            point_rows=point_rows,
            prototype_index=prototype_index,
            seed=seed,
        )
    return _build_label_visual_centers(
        point_rows=point_rows,
        prototype_index=prototype_index,
    )


def _build_kmeans_visual_centers(
    *,
    rows: Sequence[LabeledQueryRow],
    embeddings: np.ndarray,
    sample_index: np.ndarray,
    point_rows: Sequence[ProjectionPointRow],
    prototype_index: PrototypeIndex,
    seed: int,
) -> list[ProjectionOverlayRow]:
    assignment_by_index = _assign_kmeans_clusters(
        rows=rows,
        embeddings=embeddings,
        prototype_index=prototype_index,
        seed=seed,
    )
    buckets: dict[tuple[str, int], list[ProjectionPointRow]] = defaultdict(list)
    for global_index, point_row in zip(sample_index, point_rows, strict=True):
        label = str(point_row["label"])
        cluster_id = assignment_by_index[int(global_index)]
        buckets[(label, cluster_id)].append(point_row)

    visual_centers: list[ProjectionOverlayRow] = []
    for (label, cluster_id), grouped_rows in sorted(buckets.items()):
        xs = [float(row["x"]) for row in grouped_rows]
        ys = [float(row["y"]) for row in grouped_rows]
        visual_centers.append(
            {
                "label": label,
                "visual_center_id": f"{label}:kmeans_visual:{cluster_id}",
                "prototype_id": f"{label}:kmeans:{cluster_id}",
                "cluster_id": cluster_id,
                "point_count": len(grouped_rows),
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
            }
        )
    return visual_centers


def _build_label_visual_centers(
    *,
    point_rows: Sequence[ProjectionPointRow],
    prototype_index: PrototypeIndex | None,
) -> list[ProjectionOverlayRow]:
    buckets: dict[str, list[ProjectionPointRow]] = defaultdict(list)
    for point_row in point_rows:
        buckets[str(point_row["label"])].append(point_row)

    visual_centers: list[ProjectionOverlayRow] = []
    for label, grouped_rows in sorted(buckets.items()):
        xs = [float(row["x"]) for row in grouped_rows]
        ys = [float(row["y"]) for row in grouped_rows]
        prototype_id = None
        if (
            prototype_index is not None
            and label in prototype_index.categories
            and len(prototype_index.categories[label]) == 1
        ):
            prototype_id = prototype_index.categories[label][0].prototype_id
        visual_centers.append(
            {
                "label": label,
                "visual_center_id": f"{label}:visual_center",
                "prototype_id": prototype_id,
                "point_count": len(grouped_rows),
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
            }
        )
    return visual_centers


def _assign_kmeans_clusters(
    *,
    rows: Sequence[LabeledQueryRow],
    embeddings: np.ndarray,
    prototype_index: PrototypeIndex,
    seed: int,
) -> dict[int, int]:
    labels_metadata = prototype_index.metadata.get("labels", {})
    indices_by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_label[str(row["mapped_label_4"])].append(index)

    assignments: dict[int, int] = {}
    for label, indices in sorted(indices_by_label.items()):
        label_embeddings = embeddings[np.asarray(indices, dtype=np.int64)]
        label_metadata = labels_metadata.get(label, {})
        selected_k = int(
            label_metadata.get(
                "selected_k",
                max(len(prototype_index.categories.get(label, [])), 1),
            )
        )
        if selected_k <= 1 or len(indices) < 2:
            for index in indices:
                assignments[index] = 0
            continue

        fitted = KMeans(
            n_clusters=selected_k,
            random_state=seed,
            n_init="auto",
        ).fit(label_embeddings)
        for index, cluster_id in zip(indices, fitted.labels_, strict=True):
            assignments[index] = int(cluster_id)
    return assignments
