"""Query buffer selection diagnostics를 JSON/JSONL로 기록한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent.src.services.training.selection.query_buffer_selection_diagnostics import (
    QueryBufferSelectionDiagnostics,
)


@dataclass(slots=True)
class QueryBufferSelectionDiagnosticsArtifacts:
    """Selection diagnostics dump 산출물 경로."""

    candidates_path: Path
    summary_path: Path


def write_query_buffer_selection_diagnostics(
    *,
    diagnostics: QueryBufferSelectionDiagnostics,
    output_prefix: str | Path,
) -> QueryBufferSelectionDiagnosticsArtifacts:
    """Selection summary와 row trace를 파일로 저장한다."""

    prefix = Path(str(output_prefix))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    candidates_path = Path(f"{prefix}.candidates.jsonl")
    summary_path = Path(f"{prefix}.summary.json")

    with candidates_path.open("w", encoding="utf-8") as file:
        for row in diagnostics.trace_rows:
            file.write(json.dumps(row.to_mapping(), ensure_ascii=True) + "\n")

    summary_path.write_text(
        json.dumps(
            diagnostics.summary.to_mapping(),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return QueryBufferSelectionDiagnosticsArtifacts(
        candidates_path=candidates_path,
        summary_path=summary_path,
    )


__all__ = [
    "QueryBufferSelectionDiagnosticsArtifacts",
    "write_query_buffer_selection_diagnostics",
]
