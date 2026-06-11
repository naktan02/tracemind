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
    _collect_run_specs_from_dir,
    collect_run_specs,
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


def test_collect_run_specs_from_dir_discovers_report_paths_in_order(
    tmp_path: Path,
) -> None:
    method_a_report = (
        tmp_path / "method_a" / "run_20260101_000000" / "reports" / "report.json"
    )
    method_b_report = (
        tmp_path / "method_b" / "run_20260101_000000" / "reports" / "report.json"
    )
    method_a_report.parent.mkdir(parents=True)
    method_b_report.parent.mkdir(parents=True)
    method_a_report.write_text("{}\n", encoding="utf-8")
    method_b_report.write_text("{}\n", encoding="utf-8")

    specs = _collect_run_specs_from_dir(tmp_path)

    assert [spec.label for spec in specs] == ["method_a", "method_b"]
    assert [spec.report_path for spec in specs] == [method_a_report, method_b_report]


def test_collect_run_specs_from_dir_requires_reports(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing_reports"
    missing_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No reports found"):
        _collect_run_specs_from_dir(tmp_path / "missing_reports")


def test_collect_run_specs_merges_dir_and_explicit_runs(tmp_path: Path) -> None:
    method_a_report = (
        tmp_path / "method_a" / "run_20260101_000000" / "reports" / "report.json"
    )
    explicit_report = tmp_path / "explicit_report.json"
    method_a_report.parent.mkdir(parents=True)
    method_a_report.write_text("{}\n", encoding="utf-8")
    explicit_report.write_text("{}\n", encoding="utf-8")

    specs = collect_run_specs(
        explicit_runs=("supervised=" + str(explicit_report),),
        run_dirs=(str(tmp_path),),
    )

    assert [spec.label for spec in specs] == ["method_a", "supervised"]


def test_collect_run_specs_rejects_duplicate_labels(tmp_path: Path) -> None:
    method_a_report = (
        tmp_path / "method_a" / "run_20260101_000000" / "reports" / "report.json"
    )
    explicit_report = (
        tmp_path / "method_a" / "run_20260101_000001" / "reports" / "report.json"
    )
    method_a_report.parent.mkdir(parents=True, exist_ok=True)
    explicit_report.parent.mkdir(parents=True, exist_ok=True)
    method_a_report.write_text("{}\n", encoding="utf-8")
    explicit_report.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate method labels"):
        collect_run_specs(
            explicit_runs=(f"method_a={explicit_report}",),
            run_dirs=(str(tmp_path),),
        )


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
