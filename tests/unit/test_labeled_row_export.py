from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.support.query_ssl_peft.io.labeled_row_export import (
    write_labeled_row_export,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


def _row(
    query_id: str,
    label: str,
    *,
    source: str = "pseudo_label",
) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=f"text for {query_id}",
        raw_label_scheme="pseudo_label",
        raw_label=label,
        mapped_label_4=label,
        locale="ko-KR",
        annotation_source=source,
        approved_by=None,
        created_at="2026-04-12T12:00:00+00:00",
    )


def test_write_labeled_row_export_writes_jsonl_manifest_and_summary(
    tmp_path: Path,
) -> None:
    generated_at = datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc)

    artifacts = write_labeled_row_export(
        rows=[
            _row("q1", "anxiety"),
            _row("q2", "depression", source="teacher_bootstrap"),
        ],
        output_path=tmp_path / "pseudo_label_train.jsonl",
        generated_at=generated_at,
    )

    rows = load_labeled_query_rows(artifacts.jsonl_path)
    manifest = json.loads(artifacts.manifest_path.read_text())
    summary = json.loads(artifacts.summary_path.read_text())

    assert [row["query_id"] for row in rows] == ["q1", "q2"]
    assert manifest["schema_version"] == "labeled_query_row_export.v1"
    assert manifest["generated_at"] == generated_at.isoformat()
    assert manifest["label_counts"] == {"anxiety": 1, "depression": 1}
    assert summary["schema_version"] == "labeled_query_row_summary.v1"
    assert summary["unique_query_id_count"] == 2
    assert summary["annotation_source_counts"] == {
        "pseudo_label": 1,
        "teacher_bootstrap": 1,
    }
