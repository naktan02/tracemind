"""Projection과 시각화 서비스."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP

from scripts.experiments.prototype_strategy.io_utils import dump_jsonl
from scripts.experiments.prototype_strategy.models import (
    ProjectionArtifact,
    PrototypeIndex,
    PrototypeVector,
)
from scripts.experiments.prototype_strategy.sampling import sample_index_array
from scripts.labeled_query_rows import LabeledQueryRow

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(slots=True)
class ProjectionService:
    """PCA/UMAP projection 산출물을 만든다."""

    seed: int
    sample_size: int

    def create(
        self,
        *,
        rows: Sequence[LabeledQueryRow],
        embeddings: np.ndarray,
        reducers: tuple[str, ...],
        output_dir: Path,
        prototype_index: PrototypeIndex | None = None,
    ) -> tuple[ProjectionArtifact, ...]:
        sample_index = sample_index_array(
            embeddings.shape[0],
            limit=self.sample_size,
            seed=self.seed,
        )
        sampled_embeddings = embeddings[sample_index]
        sampled_rows = [rows[index] for index in sample_index]

        artifacts: list[ProjectionArtifact] = []
        for reducer_name in reducers:
            reducer = self._build_reducer(reducer_name=reducer_name)
            projection = reducer.fit_transform(sampled_embeddings)
            points_path = output_dir / f"train_{reducer_name}.jsonl"
            figure_path = output_dir / f"train_{reducer_name}.png"
            point_rows = []
            for row, point in zip(sampled_rows, projection, strict=True):
                point_rows.append(
                    {
                        "query_id": row.get("query_id", ""),
                        "label": row["mapped_label_4"],
                        "x": float(point[0]),
                        "y": float(point[1]),
                    }
                )

            prototype_points = self._project_prototypes(
                reducer=reducer,
                prototype_index=prototype_index,
            )
            prototype_points_path: Path | None = None
            if prototype_points:
                prototype_points_path = (
                    output_dir
                    / (
                        f"train_{reducer_name}."
                        f"{prototype_index.strategy_name}_prototypes.jsonl"
                    )
                )
                dump_jsonl(prototype_points_path, prototype_points)

            visual_center_points = self._build_visual_center_points(
                rows=rows,
                embeddings=embeddings,
                sample_index=sample_index,
                point_rows=point_rows,
                prototype_index=prototype_index,
            )
            visual_center_points_path: Path | None = None
            if visual_center_points:
                visual_center_suffix = (
                    prototype_index.strategy_name
                    if prototype_index is not None
                    else "label"
                )
                visual_center_points_path = (
                    output_dir
                    / (
                        f"train_{reducer_name}."
                        f"{visual_center_suffix}_visual_centers.jsonl"
                    )
                )
                dump_jsonl(visual_center_points_path, visual_center_points)

            dump_jsonl(points_path, point_rows)
            self._draw_figure(
                point_rows=point_rows,
                prototype_points=prototype_points,
                visual_center_points=visual_center_points,
                figure_path=figure_path,
                title=self._build_title(
                    reducer_name=reducer_name,
                    prototype_index=prototype_index,
                ),
            )
            artifacts.append(
                ProjectionArtifact(
                    reducer_name=reducer_name,
                    points_path=points_path,
                    figure_path=figure_path,
                    prototype_strategy_name=(
                        prototype_index.strategy_name
                        if prototype_points and prototype_index is not None
                        else None
                    ),
                    prototype_points_path=prototype_points_path,
                    visual_center_points_path=visual_center_points_path,
                )
            )
        return tuple(artifacts)

    def _build_reducer(
        self,
        *,
        reducer_name: str,
    ) -> Any:
        normalized_name = reducer_name.lower()
        if normalized_name == "pca":
            return PCA(n_components=2, random_state=self.seed)
        if normalized_name == "umap":
            return UMAP(n_components=2, random_state=self.seed)
        raise ValueError(f"Unsupported projection reducer: {reducer_name}")

    def _project_prototypes(
        self,
        *,
        reducer: Any,
        prototype_index: PrototypeIndex | None,
    ) -> list[dict[str, float | int | str]]:
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
                [
                    prototype.centroid
                    for _, prototype in indexed_prototypes
                ],
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

    def _build_title(
        self,
        *,
        reducer_name: str,
        prototype_index: PrototypeIndex | None,
    ) -> str:
        title = f"Train projection ({reducer_name.upper()})"
        if prototype_index is None:
            return title
        return f"{title} + {prototype_index.strategy_name} centroids"

    def _build_visual_center_points(
        self,
        *,
        rows: Sequence[LabeledQueryRow],
        embeddings: np.ndarray,
        sample_index: np.ndarray,
        point_rows: Sequence[dict[str, float | str]],
        prototype_index: PrototypeIndex | None,
    ) -> list[dict[str, float | int | str]]:
        if prototype_index is not None and prototype_index.strategy_name == "kmeans":
            return self._build_kmeans_visual_centers(
                rows=rows,
                embeddings=embeddings,
                sample_index=sample_index,
                point_rows=point_rows,
                prototype_index=prototype_index,
            )
        return self._build_label_visual_centers(
            point_rows=point_rows,
            prototype_index=prototype_index,
        )

    def _build_kmeans_visual_centers(
        self,
        *,
        rows: Sequence[LabeledQueryRow],
        embeddings: np.ndarray,
        sample_index: np.ndarray,
        point_rows: Sequence[dict[str, float | str]],
        prototype_index: PrototypeIndex,
    ) -> list[dict[str, float | int | str]]:
        assignment_by_index = self._assign_kmeans_clusters(
            rows=rows,
            embeddings=embeddings,
            prototype_index=prototype_index,
        )
        buckets: dict[tuple[str, int], list[dict[str, float | str]]] = defaultdict(list)
        for global_index, point_row in zip(sample_index, point_rows, strict=True):
            label = str(point_row["label"])
            cluster_id = assignment_by_index[int(global_index)]
            buckets[(label, cluster_id)].append(point_row)

        visual_centers: list[dict[str, float | int | str]] = []
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
        self,
        *,
        point_rows: Sequence[dict[str, float | str]],
        prototype_index: PrototypeIndex | None,
    ) -> list[dict[str, float | int | str]]:
        buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
        for point_row in point_rows:
            buckets[str(point_row["label"])].append(point_row)

        visual_centers: list[dict[str, float | int | str]] = []
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
        self,
        *,
        rows: Sequence[LabeledQueryRow],
        embeddings: np.ndarray,
        prototype_index: PrototypeIndex,
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
                random_state=self.seed,
                n_init="auto",
            ).fit(label_embeddings)
            for index, cluster_id in zip(indices, fitted.labels_, strict=True):
                assignments[index] = int(cluster_id)
        return assignments

    def _draw_figure(
        self,
        *,
        point_rows: Sequence[dict[str, float | str]],
        prototype_points: Sequence[dict[str, float | int | str]],
        visual_center_points: Sequence[dict[str, float | int | str]],
        figure_path: Path,
        title: str,
    ) -> None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8))
        buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in point_rows:
            buckets[str(row["label"])].append((float(row["x"]), float(row["y"])))

        for label, points in sorted(buckets.items()):
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            plt.scatter(xs, ys, s=8, alpha=0.7, label=label)

        if prototype_points:
            prototype_buckets: dict[str, list[dict[str, float | int | str]]] = (
                defaultdict(list)
            )
            for row in prototype_points:
                prototype_buckets[str(row["label"])].append(row)

            for label, rows in sorted(prototype_buckets.items()):
                xs = [float(row["x"]) for row in rows]
                ys = [float(row["y"]) for row in rows]
                plt.scatter(
                    xs,
                    ys,
                    s=180,
                    marker="X",
                    edgecolors="black",
                    linewidths=0.8,
                    label=f"{label} centroid",
                )
                for row in rows:
                    plt.annotate(
                        self._short_prototype_label(str(row["prototype_id"])),
                        (float(row["x"]), float(row["y"])),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )

        if visual_center_points:
            visual_buckets: dict[str, list[dict[str, float | int | str]]] = defaultdict(
                list
            )
            for row in visual_center_points:
                visual_buckets[str(row["label"])].append(row)

            for label, rows in sorted(visual_buckets.items()):
                xs = [float(row["x"]) for row in rows]
                ys = [float(row["y"]) for row in rows]
                plt.scatter(
                    xs,
                    ys,
                    s=110,
                    marker="D",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.1,
                    label=f"{label} visual center",
                )
                for row in rows:
                    plt.annotate(
                        self._short_visual_center_label(str(row["visual_center_id"])),
                        (float(row["x"]), float(row["y"])),
                        textcoords="offset points",
                        xytext=(5, -10),
                        fontsize=8,
                    )

        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()

    @staticmethod
    def _short_prototype_label(prototype_id: str) -> str:
        parts = prototype_id.split(":")
        if len(parts) >= 3:
            return f"{parts[0]}#{parts[-1]}"
        return parts[0]

    @staticmethod
    def _short_visual_center_label(visual_center_id: str) -> str:
        parts = visual_center_id.split(":")
        if len(parts) >= 3:
            return f"V{parts[0]}#{parts[-1]}"
        return f"V{parts[0]}"
