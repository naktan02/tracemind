"""raw CSV -> labeled query set 재사용 함수."""

from __future__ import annotations

import csv
import json
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_mapping_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def parse_created_at(
    row: dict[str, str],
    config: dict[str, Any],
) -> str:
    source = config["created_at_source"]
    default_created_at = config["default_created_at"]

    if source == "default":
        return default_created_at

    if source == "column_unix_seconds":
        column = config["created_at_column"]
        raw_value = row.get(column, "").strip()
        if not raw_value:
            return default_created_at
        return datetime.fromtimestamp(
            int(float(raw_value)),
            tz=timezone.utc,
        ).isoformat()

    if source == "column_iso8601":
        column = config["created_at_column"]
        raw_value = row.get(column, "").strip()
        return raw_value or default_created_at

    raise ValueError(f"Unsupported created_at_source: {source}")


def map_row(
    row: dict[str, str],
    *,
    row_index: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    raw_label = row[config["label_column"]].strip()
    mapped_label = config["label_map"].get(raw_label)
    if mapped_label is None:
        raise ValueError(f"Unmapped raw label '{raw_label}' in row {row_index}.")

    if mapped_label == config["exclude_if_mapped_label"]:
        return None

    query_id = row.get(config["id_column"], "").strip() or (
        f"{config['dataset_id']}::{row_index}"
    )

    return {
        "query_id": query_id,
        "text": row[config["text_column"]].strip(),
        "raw_label_scheme": config["raw_label_scheme"],
        "raw_label": raw_label,
        "mapped_label_4": mapped_label,
        "locale": config["default_locale"],
        "annotation_source": config["annotation_source"],
        "approved_by": config["approved_by"] or None,
        "created_at": parse_created_at(row, config),
    }


def build_labeled_query_set(
    *,
    raw_csv_path: Path,
    mapping_config_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    config = load_mapping_config(mapping_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = config["dataset_id"]
    jsonl_path = output_dir / f"{dataset_id}.jsonl"
    manifest_path = output_dir / f"{dataset_id}.manifest.json"

    raw_label_counts: Counter[str] = Counter()
    mapped_label_counts: Counter[str] = Counter()
    rows_seen = 0
    rows_written = 0
    rows_excluded = 0

    with raw_csv_path.open(newline="", encoding="utf-8") as source_file, jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        reader = csv.DictReader(source_file)
        for row_index, row in enumerate(reader, start=1):
            rows_seen += 1
            raw_label = row[config["label_column"]].strip()
            raw_label_counts[raw_label] += 1

            payload = map_row(row, row_index=row_index, config=config)
            if payload is None:
                rows_excluded += 1
                continue

            mapped_label_counts[payload["mapped_label_4"]] += 1
            output_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
            rows_written += 1

    manifest = {
        "schema_version": config["schema_version"],
        "dataset_id": dataset_id,
        "mapping_version": config["mapping_version"],
        "raw_label_scheme": config["raw_label_scheme"],
        "source_csv": str(raw_csv_path),
        "mapping_config": str(mapping_config_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_excluded": rows_excluded,
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "mapped_label_counts": dict(sorted(mapped_label_counts.items())),
        "output_jsonl": str(jsonl_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return jsonl_path, manifest_path
