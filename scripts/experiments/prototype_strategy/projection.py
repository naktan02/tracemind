"""Projection과 시각화 서비스."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from scripts.experiments.prototype_strategy.io_utils import dump_jsonl
from scripts.experiments.prototype_strategy.models import ProjectionArtifact
from scripts.experiments.prototype_strategy.strategies import sample_indices

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
        rows: Sequence[dict[str, str]],
        embeddings: np.ndarray,
        reducers: tuple[str, ...],
        output_dir: Path,
    ) -> tuple[ProjectionArtifact, ...]:
        sample_index = sample_indices(
            embeddings.shape[0],
            limit=self.sample_size,
            seed=self.seed,
        )
        sampled_embeddings = embeddings[sample_index]
        sampled_rows = [rows[index] for index in sample_index]

        artifacts: list[ProjectionArtifact] = []
        for reducer_name in reducers:
            projection = self._reduce(
                sampled_embeddings,
                reducer_name=reducer_name,
            )
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
            dump_jsonl(points_path, point_rows)
            self._draw_figure(
                point_rows=point_rows,
                figure_path=figure_path,
                title=f"Train projection ({reducer_name.upper()})",
            )
            artifacts.append(
                ProjectionArtifact(
                    reducer_name=reducer_name,
                    points_path=points_path,
                    figure_path=figure_path,
                )
            )
        return tuple(artifacts)

    def _reduce(
        self,
        embeddings: np.ndarray,
        *,
        reducer_name: str,
    ) -> np.ndarray:
        normalized_name = reducer_name.lower()
        if normalized_name == "pca":
            return PCA(n_components=2, random_state=self.seed).fit_transform(embeddings)
        if normalized_name == "umap":
            return UMAP(n_components=2, random_state=self.seed).fit_transform(embeddings)
        raise ValueError(f"Unsupported projection reducer: {reducer_name}")

    def _draw_figure(
        self,
        *,
        point_rows: Sequence[dict[str, float | str]],
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

        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()
