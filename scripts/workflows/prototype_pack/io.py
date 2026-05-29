"""prototype 스크립트 공통 입출력 함수."""

from __future__ import annotations

import json
from pathlib import Path

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    group_labeled_query_rows_by_label,
    load_labeled_query_rows,
)


def load_jsonl(path: Path) -> list[LabeledQueryRow]:
    return load_labeled_query_rows(path)


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_metadata_from_manifests(input_jsonl: Path) -> tuple[str, str | None]:
    direct_manifest = load_json(input_jsonl.with_suffix(".manifest.json"))
    if direct_manifest is not None and direct_manifest.get("mapping_version"):
        return direct_manifest["mapping_version"], direct_manifest.get("dataset_id")

    split_manifest_candidates: list[Path] = []
    for suffix in (".train.jsonl", ".validation.jsonl"):
        if input_jsonl.name.endswith(suffix):
            split_manifest_candidates.append(
                input_jsonl.parent / input_jsonl.name.replace(suffix, ".manifest.json")
            )
    for manifest_path in split_manifest_candidates:
        split_manifest = load_json(manifest_path)
        if split_manifest is not None and split_manifest.get("source_mapping_version"):
            return (
                split_manifest["source_mapping_version"],
                split_manifest.get("source_dataset_id"),
            )

    raise ValueError(
        "Could not resolve mapping_version from manifests. "
        "Generate the labeled_query_set manifest or split manifest first."
    )


def group_rows_by_label(
    rows: list[LabeledQueryRow],
) -> dict[str, list[LabeledQueryRow]]:
    return group_labeled_query_rows_by_label(rows)
