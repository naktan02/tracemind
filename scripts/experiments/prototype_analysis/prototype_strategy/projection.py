"""Projection과 시각화 서비스."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from methods.prototype.index import PrototypeIndex
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

from .io_utils import dump_jsonl
from .models import ProjectionArtifact
from .projection_figure import draw_projection_figure
from .projection_rows import (
    build_projected_point_rows,
    build_visual_center_points,
    project_prototypes,
)
from .sampling import sample_index_array


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
            point_rows = build_projected_point_rows(
                rows=sampled_rows,
                projection=projection,
            )

            prototype_points = project_prototypes(
                reducer=reducer,
                prototype_index=prototype_index,
            )
            prototype_points_path: Path | None = None
            if prototype_points:
                prototype_points_path = output_dir / (
                    f"train_{reducer_name}."
                    f"{prototype_index.strategy_name}_prototypes.jsonl"
                )
                dump_jsonl(prototype_points_path, prototype_points)

            visual_center_points = build_visual_center_points(
                rows=rows,
                embeddings=embeddings,
                sample_index=sample_index,
                point_rows=point_rows,
                prototype_index=prototype_index,
                seed=self.seed,
            )
            visual_center_points_path: Path | None = None
            if visual_center_points:
                visual_center_suffix = (
                    prototype_index.strategy_name
                    if prototype_index is not None
                    else "label"
                )
                visual_center_points_path = output_dir / (
                    f"train_{reducer_name}.{visual_center_suffix}_visual_centers.jsonl"
                )
                dump_jsonl(visual_center_points_path, visual_center_points)

            dump_jsonl(points_path, point_rows)
            draw_projection_figure(
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
