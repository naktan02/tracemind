from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from methods.adaptation.text_encoder_classifier.projection import reduce_features_2d
from scripts.experiments.central.ssl_control.build_method_projection_figure import (
    MethodFeatureSet,
    draw_method_figure,
    parse_run_specs,
    resolve_projection_output_dir,
    select_row_indices,
    write_split_projection_artifacts,
)


def test_parse_run_specs_requires_label_and_existing_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")

    specs = parse_run_specs([f"fixmatch={report_path}"])

    assert specs[0].label == "fixmatch"
    assert specs[0].report_path == report_path

    with pytest.raises(ValueError, match="label=path"):
        parse_run_specs([str(report_path)])


def test_select_row_indices_is_reproducible_and_keeps_order() -> None:
    first = select_row_indices(row_count=100, max_rows=10, seed=42, salt="a")
    second = select_row_indices(row_count=100, max_rows=10, seed=42, salt="a")

    assert first == second
    assert first == sorted(first)
    assert len(first) == 10


def test_reduce_features_2d_supports_pca() -> None:
    features = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    projection = reduce_features_2d(
        features=features,
        reducer_name="pca",
        seed=42,
        n_neighbors=15,
    )

    assert projection.reducer == "pca"
    assert projection.fallback_reason is None
    assert projection.coordinates.shape == (4, 2)


def test_resolve_projection_output_dir_uses_dated_child(tmp_path: Path) -> None:
    created_at = datetime(2026, 6, 4, 1, 2, 3)
    output_root = tmp_path / "method_projection"

    output_dir = resolve_projection_output_dir(
        output_dir=None,
        output_root=output_root,
        figure_version=None,
        created_at=created_at,
    )

    assert output_dir == output_root / "2026_06_04_010203"
    assert (
        resolve_projection_output_dir(
            output_dir=tmp_path / "exact",
            output_root=output_root,
            figure_version="ignored",
            created_at=created_at,
        )
        == tmp_path / "exact"
    )
    assert (
        resolve_projection_output_dir(
            output_dir=None,
            output_root=output_root,
            figure_version="paper_v1",
            created_at=created_at,
        )
        == output_root / "2026_06_04_010203_paper_v1"
    )


def test_write_split_projection_artifacts_writes_method_files(
    tmp_path: Path,
) -> None:
    feature_sets = [
        MethodFeatureSet(
            method_label="supervised",
            trainer_version="peft_clf_1",
            split="test",
            features=np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
            row_ids=("q1", "q2"),
            true_labels=("anxiety", "normal"),
            predicted_labels=("anxiety", "normal"),
            top_1_probabilities=(0.9, 0.8),
        ),
        MethodFeatureSet(
            method_label="fixmatch",
            trainer_version="peft_fixmatch_1",
            split="test",
            features=np.asarray([[1.0, 1.0], [1.1, 1.0]], dtype=np.float32),
            row_ids=("q1", "q2"),
            true_labels=("anxiety", "normal"),
            predicted_labels=("normal", "normal"),
            top_1_probabilities=(0.7, 0.95),
        ),
    ]

    outputs = write_split_projection_artifacts(
        split="test",
        feature_sets=feature_sets,
        categories=["anxiety", "normal"],
        output_dir=tmp_path,
        reducer_name="pca",
        seed=42,
        n_neighbors=15,
    )

    assert outputs["reducer"] == "pca"
    assert outputs["method_count"] == 2
    assert set(outputs["methods"]) == {"supervised", "fixmatch"}
    assert not (tmp_path / "test.method_projection.png").exists()
    assert not (tmp_path / "test.method_projection.jsonl").exists()

    points_path = Path(outputs["methods"]["fixmatch"]["points_jsonl"])
    rows = [
        json.loads(line)
        for line in points_path.read_text(encoding="utf-8").splitlines()
    ]

    assert Path(outputs["methods"]["supervised"]["figure_png"]).exists()
    assert Path(outputs["methods"]["supervised"]["features_npz"]).exists()
    assert Path(outputs["methods"]["fixmatch"]["figure_png"]).exists()
    assert Path(outputs["methods"]["fixmatch"]["features_npz"]).exists()
    assert len(rows) == 2
    assert {row["method_label"] for row in rows} == {"fixmatch"}
    assert any(not row["is_correct"] for row in rows)


def test_draw_method_figure_does_not_mark_incorrect_by_default(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "x": index * 0.01,
            "y": index * 0.01,
            "label": "anxiety",
            "predicted_label": "normal",
            "is_correct": False,
            "top_1_probability": 0.7,
        }
        for index in range(80)
    ]
    figure_path = tmp_path / "default.png"

    draw_method_figure(
        figure_path=figure_path,
        rows=rows,
        categories=["anxiety", "normal"],
        title="test",
        axis_limits=((-0.1, 1.0), (-0.1, 1.0)),
    )

    pixels = np.asarray(Image.open(figure_path).convert("RGB"))
    black_pixels = int(np.all(pixels < 20, axis=2).sum())

    assert black_pixels / pixels.shape[0] / pixels.shape[1] < 0.015
