"""Multiview query adaptation dataset을 labeled row JSONL로 내보낸다."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from scripts.support.query_ssl_text_encoder.io.artifact_writer import (
    write_json_artifact,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)

QUERY_ADAPTATION_MULTIVIEW_EXPORT_SCHEMA_VERSION = (
    "query_adaptation_multiview_export.v1"
)
QUERY_ADAPTATION_MULTIVIEW_SUMMARY_SCHEMA_VERSION = (
    "query_adaptation_multiview_summary.v1"
)


@dataclass(slots=True)
class QueryAdaptationMultiviewExportArtifacts:
    """Multiview export 산출물 경로."""

    jsonl_path: Path
    manifest_path: Path
    summary_path: Path


def build_labeled_rows_from_query_adaptation_multiview_dataset(
    dataset: Any,
    *,
    annotation_source: str = "query_adaptation_multiview",
) -> list[LabeledQueryRow]:
    """Multiview adaptation dataset을 scripts용 labeled row shape로 변환한다."""

    rows: list[LabeledQueryRow] = []
    for example in dataset.examples:
        base_example = example.base_example
        source_row = example.source_row
        approved_by = (
            None if base_example.label_source == "pseudo_label" else "manual_override"
        )
        row: LabeledQueryRow = {
            "query_id": base_example.query_id,
            "text": source_row.text,
            "raw_label_scheme": base_example.label_source,
            "raw_label": base_example.label,
            "mapped_label_4": base_example.label,
            "locale": base_example.provenance.locale,
            "annotation_source": annotation_source,
            "approved_by": approved_by,
            "created_at": source_row.occurred_at.isoformat(),
            "weak_text": cast(str, source_row.weak_text),
            "strong_text": cast(str, source_row.strong_text),
        }
        if source_row.weak_translated_text is not None:
            row["weak_translated_text"] = source_row.weak_translated_text
        if source_row.strong_translated_text is not None:
            row["strong_translated_text"] = source_row.strong_translated_text
        rows.append(row)
    return rows


def write_query_adaptation_multiview_dataset(
    *,
    dataset: Any,
    output_path: str | Path,
    annotation_source: str = "query_adaptation_multiview",
    generated_at: datetime | None = None,
) -> QueryAdaptationMultiviewExportArtifacts:
    """Multiview labeled row JSONL과 요약 산출물을 기록한다."""

    rows = build_labeled_rows_from_query_adaptation_multiview_dataset(
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
    write_json_artifact(
        path=manifest_path,
        payload={
            "schema_version": QUERY_ADAPTATION_MULTIVIEW_EXPORT_SCHEMA_VERSION,
            "generated_at": effective_generated_at.isoformat(),
            "annotation_source": annotation_source,
            "row_count": len(rows),
            "label_counts": dict(
                sorted(Counter(row["mapped_label_4"] for row in rows).items())
            ),
            "label_source_counts": dict(
                sorted(Counter(row["raw_label_scheme"] for row in rows).items())
            ),
        },
    )
    write_json_artifact(
        path=summary_path,
        payload=_build_query_adaptation_multiview_summary(
            dataset=dataset,
            annotation_source=annotation_source,
            generated_at=effective_generated_at,
        ),
    )
    return QueryAdaptationMultiviewExportArtifacts(
        jsonl_path=resolved_output_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def _build_query_adaptation_multiview_summary(
    *,
    dataset: Any,
    annotation_source: str,
    generated_at: datetime,
) -> dict[str, object]:
    augmenter_name_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    weak_translated_counts: Counter[str] = Counter()
    strong_translated_counts: Counter[str] = Counter()

    for example in dataset.examples:
        base_example = example.base_example
        views = example.views
        augmenter_name_counts[views.augmenter_name] += 1
        label_counts[base_example.label] += 1
        label_source_counts[base_example.label_source] += 1
        locale_counts[base_example.provenance.locale] += 1
        weak_translated_counts[str(views.weak_translated_text is not None).lower()] += 1
        strong_translated_counts[
            str(views.strong_translated_text is not None).lower()
        ] += 1

    return {
        "schema_version": QUERY_ADAPTATION_MULTIVIEW_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "annotation_source": annotation_source,
        "row_count": dataset.count,
        "augmenter_name_counts": dict(sorted(augmenter_name_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "locale_counts": dict(sorted(locale_counts.items())),
        "weak_translated_text_present_counts": dict(
            sorted(weak_translated_counts.items())
        ),
        "strong_translated_text_present_counts": dict(
            sorted(strong_translated_counts.items())
        ),
    }
