"""Teacher pseudo-label artifact writer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


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
        _write_jsonl(prediction_trace_path, prediction_trace_rows)
        _write_json(prediction_summary_path, prediction_summary)
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
        _write_json(bootstrap_summary_path, bootstrap_summary)
        return bootstrap_summary_path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=True) + "\n")
