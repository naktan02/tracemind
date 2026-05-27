"""Query adaptation dataset을 PEFT baseline JSONL 입력으로 내보내는 helper."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import fsum
from pathlib import Path
from typing import Any

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)

QUERY_ADAPTATION_EXPORT_SCHEMA_VERSION = "query_adaptation_lora_export.v1"
QUERY_ADAPTATION_SUMMARY_SCHEMA_VERSION = "query_adaptation_dataset_summary.v1"


@dataclass(slots=True)
class QueryAdaptationLoraExportArtifacts:
    """query adaptation dataset export 산출물 경로."""

    jsonl_path: Path
    manifest_path: Path
    summary_path: Path


def build_labeled_rows_from_query_adaptation_dataset(
    dataset: Any,
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
    dataset: Any,
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
    summary_path = resolved_output_path.with_suffix(
        f"{resolved_output_path.suffix}.summary.json"
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
    summary_path.write_text(
        json.dumps(
            _build_query_adaptation_summary(
                dataset=dataset,
                annotation_source=annotation_source,
                generated_at=effective_generated_at,
            ),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return QueryAdaptationLoraExportArtifacts(
        jsonl_path=resolved_output_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def _build_query_adaptation_summary(
    *,
    dataset: Any,
    annotation_source: str,
    generated_at: datetime,
) -> dict[str, object]:
    selection_stage_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    model_revision_counts: Counter[str] = Counter()
    confidence_kind_counts: Counter[str] = Counter()
    translated_text_present_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    margin_values: list[float] = []

    for example in dataset.examples:
        provenance = example.provenance
        locale_counts[provenance.locale] += 1
        source_type_counts[provenance.source_type] += 1
        model_revision_counts[provenance.model_revision] += 1
        confidence_kind_counts[provenance.selection_confidence_kind] += 1
        translated_text_present_counts[
            str(provenance.translated_text_present).lower()
        ] += 1
        selection_context = provenance.selection_context
        if selection_context is not None:
            selection_stage_counts[selection_context.selection_stage.value] += 1
        confidence_values.append(float(example.confidence))
        margin_values.append(float(example.margin))

    return {
        "schema_version": QUERY_ADAPTATION_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "annotation_source": annotation_source,
        "row_count": dataset.count,
        "label_counts": dict(
            sorted(Counter(example.label for example in dataset.examples).items())
        ),
        "label_source_counts": dict(
            sorted(
                Counter(example.label_source for example in dataset.examples).items()
            )
        ),
        "locale_counts": dict(sorted(locale_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "model_revision_counts": dict(sorted(model_revision_counts.items())),
        "selection_confidence_kind_counts": dict(
            sorted(confidence_kind_counts.items())
        ),
        "translated_text_present_counts": dict(
            sorted(translated_text_present_counts.items())
        ),
        "selection_stage_counts": dict(sorted(selection_stage_counts.items())),
        "confidence_stats": _summarize_scalar_values(confidence_values),
        "margin_stats": _summarize_scalar_values(margin_values),
    }


def _summarize_scalar_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": fsum(values) / len(values),
    }
