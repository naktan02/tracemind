"""Query adaptation dataset을 LoRA baseline JSONL 입력으로 내보내는 helper."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.src.services.training.query_adaptation_dataset_service import (
    QueryAdaptationDataset,
)
from scripts.labeled_query_rows import LabeledQueryRow, dump_labeled_query_rows

QUERY_ADAPTATION_EXPORT_SCHEMA_VERSION = "query_adaptation_lora_export.v1"


@dataclass(slots=True)
class QueryAdaptationLoraExportArtifacts:
    """query adaptation dataset export 산출물 경로."""

    jsonl_path: Path
    manifest_path: Path


def build_labeled_rows_from_query_adaptation_dataset(
    dataset: QueryAdaptationDataset,
    *,
    annotation_source: str = "query_adaptation_dataset",
) -> list[LabeledQueryRow]:
    """agent-local adaptation dataset을 scripts용 labeled row shape로 변환한다."""

    rows: list[LabeledQueryRow] = []
    for example in dataset.examples:
        approved_by = (
            None if example.label_source == "pseudo_label" else "manual_override"
        )
        rows.append(
            LabeledQueryRow(
                query_id=example.query_id,
                text=example.source_row.text,
                raw_label_scheme=example.label_source,
                raw_label=example.label,
                mapped_label_4=example.label,
                locale=example.provenance.locale,
                annotation_source=annotation_source,
                approved_by=approved_by,
                created_at=example.source_row.occurred_at.isoformat(),
            )
        )
    return rows


def write_query_adaptation_lora_dataset(
    *,
    dataset: QueryAdaptationDataset,
    output_path: str | Path,
    annotation_source: str = "query_adaptation_dataset",
    generated_at: datetime | None = None,
) -> QueryAdaptationLoraExportArtifacts:
    """LoRA supervised baseline이 읽을 JSONL과 manifest를 기록한다."""

    rows = build_labeled_rows_from_query_adaptation_dataset(
        dataset,
        annotation_source=annotation_source,
    )
    resolved_output_path = Path(str(output_path))
    dump_labeled_query_rows(resolved_output_path, rows)

    effective_generated_at = generated_at or datetime.now(tz=timezone.utc)
    manifest_path = resolved_output_path.with_suffix(
        f"{resolved_output_path.suffix}.manifest.json"
    )
    manifest = {
        "schema_version": QUERY_ADAPTATION_EXPORT_SCHEMA_VERSION,
        "generated_at": effective_generated_at.isoformat(),
        "annotation_source": annotation_source,
        "row_count": len(rows),
        "label_counts": dict(
            sorted(Counter(row["mapped_label_4"] for row in rows).items())
        ),
        "label_source_counts": dict(
            sorted(Counter(row["raw_label_scheme"] for row in rows).items())
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return QueryAdaptationLoraExportArtifacts(
        jsonl_path=resolved_output_path,
        manifest_path=manifest_path,
    )


__all__ = [
    "QUERY_ADAPTATION_EXPORT_SCHEMA_VERSION",
    "QueryAdaptationLoraExportArtifacts",
    "build_labeled_rows_from_query_adaptation_dataset",
    "write_query_adaptation_lora_dataset",
]
