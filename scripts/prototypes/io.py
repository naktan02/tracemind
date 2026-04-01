"""prototype 스크립트 공통 입출력 함수."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any] | None:
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


def group_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["mapped_label_4"]].append(row)
    return dict(sorted(rows_by_label.items()))
