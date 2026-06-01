"""Teacher pseudo-label artifact writer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.support.query_ssl_text_encoder.io.artifact_writer import (
    write_json_artifact,
    write_jsonl_artifact,
)


@dataclass(frozen=True, slots=True)
class TeacherPseudoLabelArtifactPaths:
    """Teacher pseudo-label export artifact 경로."""

    prediction_trace_jsonl: Path
    prediction_summary_json: Path


class TeacherPseudoLabelArtifactWriter:
    """teacher prediction trace와 bootstrap summary 파일 저장만 담당한다."""

    def write_prediction_artifacts(
        self,
        *,
        export_dir: Path,
        prediction_trace_rows: Sequence[Mapping[str, object]],
        prediction_summary: Mapping[str, object],
    ) -> TeacherPseudoLabelArtifactPaths:
        prediction_trace_path = export_dir / "teacher_unlabeled_predictions.jsonl"
        prediction_summary_path = (
            export_dir / "teacher_unlabeled_predictions.summary.json"
        )
        write_jsonl_artifact(
            path=prediction_trace_path,
            rows=prediction_trace_rows,
        )
        write_json_artifact(
            path=prediction_summary_path,
            payload=dict(prediction_summary),
        )
        return TeacherPseudoLabelArtifactPaths(
            prediction_trace_jsonl=prediction_trace_path,
            prediction_summary_json=prediction_summary_path,
        )

    def write_bootstrap_summary(
        self,
        *,
        export_dir: Path,
        bootstrap_summary: Mapping[str, object],
    ) -> Path:
        bootstrap_summary_path = export_dir / "bootstrap.summary.json"
        write_json_artifact(
            path=bootstrap_summary_path,
            payload=dict(bootstrap_summary),
        )
        return bootstrap_summary_path
