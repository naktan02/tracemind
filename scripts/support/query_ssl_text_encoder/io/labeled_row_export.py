"""LabeledQueryRow JSONL export와 metadata writer."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)

LABELED_ROW_EXPORT_SCHEMA_VERSION = "labeled_query_row_export.v1"
LABELED_ROW_SUMMARY_SCHEMA_VERSION = "labeled_query_row_summary.v1"


@dataclass(slots=True)
class LabeledRowExportArtifacts:
    """Labeled row JSONL export 산출물."""

    jsonl_path: Path
    manifest_path: Path
    summary_path: Path


def write_labeled_row_export(
    *,
    rows: Sequence[LabeledQueryRow],
    output_path: str | Path,
    generated_at: datetime,
) -> LabeledRowExportArtifacts:
    """labeled rows와 manifest/summary metadata를 함께 저장한다."""

    resolved_output_path = Path(str(output_path))
    dump_labeled_query_rows(resolved_output_path, rows)
    manifest_path = resolved_output_path.with_suffix(
        f"{resolved_output_path.suffix}.manifest.json"
    )
    summary_path = resolved_output_path.with_suffix(
        f"{resolved_output_path.suffix}.summary.json"
    )
    manifest = {
        "schema_version": LABELED_ROW_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "row_count": len(rows),
        "label_counts": dict(
            sorted(Counter(str(row["mapped_label_4"]) for row in rows).items())
        ),
        "raw_label_scheme_counts": dict(
            sorted(Counter(str(row["raw_label_scheme"]) for row in rows).items())
        ),
    }
    summary = {
        "schema_version": LABELED_ROW_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "row_count": len(rows),
        "unique_query_id_count": len({str(row["query_id"]) for row in rows}),
        "annotation_source_counts": dict(
            sorted(Counter(str(row["annotation_source"]) for row in rows).items())
        ),
        "approved_by_counts": dict(
            sorted(
                Counter(
                    "none" if row["approved_by"] is None else str(row["approved_by"])
                    for row in rows
                ).items()
            )
        ),
        "locale_counts": dict(
            sorted(Counter(str(row["locale"]) for row in rows).items())
        ),
        "label_counts": manifest["label_counts"],
        "raw_label_scheme_counts": manifest["raw_label_scheme_counts"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return LabeledRowExportArtifacts(
        jsonl_path=resolved_output_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
