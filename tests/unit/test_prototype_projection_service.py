"""Prototype projection service unit tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from methods.prototype.index import (
    PrototypeIndex,
    PrototypeVector,
)
from scripts.experiments.prototype_analysis.prototype_strategy.projection import (
    ProjectionService,
)


def _row(query_id: str, label: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "text": query_id,
        "mapped_label_4": label,
    }


def test_projection_service_writes_projected_kmeans_centroids(
    tmp_path,
) -> None:
    service = ProjectionService(seed=42, sample_size=10)
    rows = [
        _row("a1", "anxiety"),
        _row("a2", "anxiety"),
        _row("n1", "normal"),
        _row("n2", "normal"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float64,
    )
    prototype_index = PrototypeIndex(
        strategy_name="kmeans",
        metadata={
            "labels": {
                "anxiety": {"selected_k": 2},
                "normal": {"selected_k": 1},
            }
        },
        categories={
            "anxiety": [
                PrototypeVector(
                    prototype_id="anxiety:kmeans:0",
                    centroid=[1.0, 0.0],
                    member_count=1,
                ),
                PrototypeVector(
                    prototype_id="anxiety:kmeans:1",
                    centroid=[0.9, 0.1],
                    member_count=1,
                ),
            ],
            "normal": [
                PrototypeVector(
                    prototype_id="normal:kmeans:0",
                    centroid=[-1.0, 0.0],
                    member_count=2,
                )
            ],
        },
    )

    artifacts = service.create(
        rows=rows,
        embeddings=embeddings,
        reducers=("pca",),
        output_dir=tmp_path,
        prototype_index=prototype_index,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.prototype_strategy_name == "kmeans"
    assert artifact.figure_path.exists()
    assert artifact.points_path.exists()
    assert artifact.prototype_points_path is not None
    assert artifact.prototype_points_path.exists()
    assert artifact.visual_center_points_path is not None
    assert artifact.visual_center_points_path.exists()

    prototype_rows = [
        json.loads(line)
        for line in artifact.prototype_points_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(prototype_rows) == 3
    assert {row["label"] for row in prototype_rows} == {"anxiety", "normal"}
    assert {row["prototype_id"] for row in prototype_rows} == {
        "anxiety:kmeans:0",
        "anxiety:kmeans:1",
        "normal:kmeans:0",
    }

    visual_center_rows = [
        json.loads(line)
        for line in artifact.visual_center_points_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(visual_center_rows) == 3
    assert {row["prototype_id"] for row in visual_center_rows} == {
        "anxiety:kmeans:0",
        "anxiety:kmeans:1",
        "normal:kmeans:0",
    }


def test_projection_service_writes_single_visual_centers_from_projected_points(
    tmp_path,
) -> None:
    service = ProjectionService(seed=42, sample_size=10)
    rows = [
        _row("a1", "anxiety"),
        _row("a2", "anxiety"),
        _row("n1", "normal"),
        _row("n2", "normal"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, -0.2],
        ],
        dtype=np.float64,
    )
    prototype_index = PrototypeIndex(
        strategy_name="single",
        categories={
            "anxiety": [
                PrototypeVector(
                    prototype_id="anxiety:single",
                    centroid=[0.9, 0.1],
                    member_count=2,
                )
            ],
            "normal": [
                PrototypeVector(
                    prototype_id="normal:single",
                    centroid=[-0.9, -0.1],
                    member_count=2,
                )
            ],
        },
    )

    artifacts = service.create(
        rows=rows,
        embeddings=embeddings,
        reducers=("pca",),
        output_dir=tmp_path,
        prototype_index=prototype_index,
    )

    artifact = artifacts[0]
    point_rows = [
        json.loads(line)
        for line in artifact.points_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    visual_center_rows = [
        json.loads(line)
        for line in artifact.visual_center_points_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    expected_by_label: dict[str, tuple[float, float]] = {}
    for label in {"anxiety", "normal"}:
        label_rows = [row for row in point_rows if row["label"] == label]
        expected_by_label[label] = (
            sum(float(row["x"]) for row in label_rows) / len(label_rows),
            sum(float(row["y"]) for row in label_rows) / len(label_rows),
        )

    for row in visual_center_rows:
        expected_x, expected_y = expected_by_label[row["label"]]
        assert float(row["x"]) == pytest.approx(expected_x)
        assert float(row["y"]) == pytest.approx(expected_y)
