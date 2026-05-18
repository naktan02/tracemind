"""Progress JSON helpers for Query SSL view materialization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from scripts.datasets.lib.query_ssl_view_models import (
    QUERY_SSL_VIEWS_PROGRESS_SCHEMA_VERSION,
    QuerySslViewArtifacts,
    ViewPartitionArtifacts,
)


def build_initial_progress(
    *,
    split_dir: Path,
    split_name: str,
    augmenter_name: str,
    augmenter_manifest: dict[str, object],
    chunk_size: int,
    artifacts: QuerySslViewArtifacts,
    partitions: Sequence[ViewPartitionArtifacts],
) -> dict[str, object]:
    return {
        "schema_version": QUERY_SSL_VIEWS_PROGRESS_SCHEMA_VERSION,
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "split_name": split_name,
        "source_split_dir": str(split_dir),
        "augmenter_name": augmenter_name,
        "augmenter": augmenter_manifest,
        "chunk_size": chunk_size,
        "partitions": {
            partition.partition_name: {
                "status": "pending",
                "source_jsonl": str(partition.source_jsonl),
                "tmp_jsonl": str(partition.tmp_jsonl),
                "final_jsonl": str(partition.final_jsonl),
                "total_count": None,
                "processed_count": 0,
                "progress_ratio": 0.0,
            }
            for partition in partitions
        },
        "manifest_json": str(artifacts.manifest_json),
        "summary_json": str(artifacts.summary_json),
    }


def update_partition_progress(
    *,
    progress: dict[str, object],
    partition: ViewPartitionArtifacts,
    total_count: int,
    processed_count: int,
    status: str,
) -> None:
    partitions = progress["partitions"]
    if not isinstance(partitions, dict):
        raise TypeError("progress['partitions'] must be a mapping.")
    partition_progress = partitions[partition.partition_name]
    if not isinstance(partition_progress, dict):
        raise TypeError("partition progress must be a mapping.")
    partition_progress["status"] = status
    partition_progress["total_count"] = total_count
    partition_progress["processed_count"] = processed_count
    partition_progress["progress_ratio"] = (
        0.0 if total_count == 0 else (processed_count / total_count)
    )
    progress["updated_at"] = utc_now()


def write_progress(path: Path, progress: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(progress, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
