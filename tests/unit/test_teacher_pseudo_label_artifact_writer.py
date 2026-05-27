"""Teacher pseudo-label artifact writer 검증."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.query_peft_ssl.io import (
    teacher_pseudo_label_artifact_writer,
)


def test_teacher_pseudo_label_artifact_writer_writes_trace_and_summaries(
    tmp_path: Path,
) -> None:
    writer = teacher_pseudo_label_artifact_writer.TeacherPseudoLabelArtifactWriter()
    trace_rows = [
        {"query_id": "q1", "predicted_label": "normal", "confidence": 0.91},
        {"query_id": "q2", "predicted_label": "anxiety", "confidence": 0.72},
    ]
    prediction_summary = {
        "schema_version": "fixed_classifier_teacher_summary.v1",
        "accepted_count": 1,
    }
    bootstrap_summary = {
        "schema_version": "fixed_classifier_lora_bootstrap.v1",
        "bootstrap_version": "bootstrap_v1",
    }

    paths = writer.write_prediction_artifacts(
        export_dir=tmp_path,
        prediction_trace_rows=trace_rows,
        prediction_summary=prediction_summary,
    )
    bootstrap_summary_path = writer.write_bootstrap_summary(
        export_dir=tmp_path,
        bootstrap_summary=bootstrap_summary,
    )

    assert paths.prediction_trace_jsonl == (
        tmp_path / "teacher_unlabeled_predictions.jsonl"
    )
    assert paths.prediction_summary_json == (
        tmp_path / "teacher_unlabeled_predictions.summary.json"
    )
    assert bootstrap_summary_path == tmp_path / "bootstrap.summary.json"
    assert [
        json.loads(line)
        for line in paths.prediction_trace_jsonl.read_text(
            encoding="utf-8"
        ).splitlines()
    ] == trace_rows
    assert (
        json.loads(paths.prediction_summary_json.read_text(encoding="utf-8"))
        == prediction_summary
    )
    assert (
        json.loads(bootstrap_summary_path.read_text(encoding="utf-8"))
        == bootstrap_summary
    )
