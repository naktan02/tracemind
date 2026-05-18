"""Query SSL view materialization manifest builders."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from scripts.datasets.lib.query_ssl_view_models import (
    QUERY_SSL_VIEWS_SCHEMA_VERSION,
    QuerySslViewArtifacts,
    ViewPartitionResult,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    count_labeled_query_rows_by_label,
)


def build_query_ssl_views_manifest(
    *,
    split_dir: Path,
    split_name: str,
    augmenter_name: str,
    augmenter_manifest: dict[str, object],
    chunk_size: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    partition_results: Sequence[ViewPartitionResult],
    artifacts: QuerySslViewArtifacts,
) -> dict[str, object]:
    return {
        "schema_version": QUERY_SSL_VIEWS_SCHEMA_VERSION,
        "split_name": split_name,
        "source_split_dir": str(split_dir),
        "source_split_manifest_json": str(split_dir / "manifest.json"),
        "augmenter_name": augmenter_name,
        "augmenter": augmenter_manifest,
        "chunk_size": chunk_size,
        "weak_view_policy": {
            "labeled": "text",
            "unlabeled": "text",
        },
        "strong_view_policy": {
            "labeled": ["aug_0", "aug_1"],
            "unlabeled": ["aug_0", "aug_1"],
        },
        "row_counts": {
            "labeled_train": len(labeled_rows),
            "unlabeled_pool": len(unlabeled_rows),
        },
        "label_counts": {
            "labeled_train": count_labeled_query_rows_by_label(labeled_rows),
            "unlabeled_pool": count_labeled_query_rows_by_label(unlabeled_rows),
        },
        "view_counts": {
            "labeled_train": count_view_metadata(labeled_rows),
            "unlabeled_pool": count_view_metadata(unlabeled_rows),
        },
        "resume": {
            result.partition_name: {
                "resumed_from_count": result.resumed_from_count,
            }
            for result in partition_results
        },
        "artifacts": {
            "labeled_train_with_views_jsonl": str(
                artifacts.labeled_train_with_views_jsonl
            ),
            "unlabeled_pool_with_views_jsonl": str(
                artifacts.unlabeled_pool_with_views_jsonl
            ),
            "manifest_json": str(artifacts.manifest_json),
            "summary_json": str(artifacts.summary_json),
            "progress_json": str(artifacts.progress_json),
        },
    }


def build_query_ssl_views_summary(
    manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": manifest["schema_version"],
        "split_name": manifest["split_name"],
        "augmenter_name": manifest["augmenter_name"],
        "chunk_size": manifest["chunk_size"],
        "row_counts": manifest["row_counts"],
        "label_counts": manifest["label_counts"],
        "view_counts": manifest["view_counts"],
        "resume": manifest["resume"],
    }


def count_view_metadata(rows: Sequence[LabeledQueryRow]) -> dict[str, object]:
    aug_0_pivot_counts: Counter[str] = Counter()
    aug_1_pivot_counts: Counter[str] = Counter()
    empty_aug_0_count = 0
    empty_aug_1_count = 0
    for row in rows:
        aug_0_pivot_counts[str(row.get("aug_0_pivot_lang", ""))] += 1
        aug_1_pivot_counts[str(row.get("aug_1_pivot_lang", ""))] += 1
        if not str(row.get("aug_0", "")).strip():
            empty_aug_0_count += 1
        if not str(row.get("aug_1", "")).strip():
            empty_aug_1_count += 1

    return {
        "aug_0_pivot_lang_counts": dict(sorted(aug_0_pivot_counts.items())),
        "aug_1_pivot_lang_counts": dict(sorted(aug_1_pivot_counts.items())),
        "empty_aug_0_count": empty_aug_0_count,
        "empty_aug_1_count": empty_aug_1_count,
    }
