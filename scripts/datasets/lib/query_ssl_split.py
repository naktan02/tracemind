"""중앙 Query SSL용 labeled/unlabeled split materialization."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    count_labeled_query_rows_by_label,
    dump_labeled_query_rows,
    group_labeled_query_rows_by_label,
    load_labeled_query_rows,
)

QUERY_SSL_SPLIT_SCHEMA_VERSION = "query_ssl_split.v1"


@dataclass(frozen=True, slots=True)
class QuerySslSplitArtifacts:
    """Query SSL split materializer가 쓰는 산출물 경로."""

    labeled_train_jsonl: Path
    unlabeled_pool_jsonl: Path
    validation_jsonl: Path
    test_jsonl: Path
    manifest_json: Path
    summary_json: Path


def materialize_class_balanced_query_ssl_split(
    *,
    source_train_jsonl: Path,
    source_validation_jsonl: Path,
    source_test_jsonl: Path,
    split_name: str,
    labeled_count_per_class: int,
    seed: int,
    output_root: Path,
) -> QuerySslSplitArtifacts:
    """labeled는 class-balanced, unlabeled는 나머지 전체로 split한다."""

    if labeled_count_per_class <= 0:
        raise ValueError("labeled_count_per_class must be positive.")
    normalized_split_name = split_name.strip()
    if not normalized_split_name:
        raise ValueError("split_name must not be empty.")

    source_train_rows = load_labeled_query_rows(source_train_jsonl)
    validation_rows = load_labeled_query_rows(source_validation_jsonl)
    test_rows = load_labeled_query_rows(source_test_jsonl)
    labeled_rows, unlabeled_rows = _split_labeled_and_unlabeled_rows(
        rows=source_train_rows,
        labeled_count_per_class=labeled_count_per_class,
        seed=seed,
    )

    output_dir = output_root / normalized_split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = QuerySslSplitArtifacts(
        labeled_train_jsonl=output_dir / "labeled_train.jsonl",
        unlabeled_pool_jsonl=output_dir / "unlabeled_pool.jsonl",
        validation_jsonl=output_dir / "validation.jsonl",
        test_jsonl=output_dir / "test.jsonl",
        manifest_json=output_dir / "manifest.json",
        summary_json=output_dir / "summary.json",
    )

    dump_labeled_query_rows(artifacts.labeled_train_jsonl, labeled_rows)
    dump_labeled_query_rows(artifacts.unlabeled_pool_jsonl, unlabeled_rows)
    dump_labeled_query_rows(artifacts.validation_jsonl, validation_rows)
    dump_labeled_query_rows(artifacts.test_jsonl, test_rows)

    manifest = _build_query_ssl_split_manifest(
        split_name=normalized_split_name,
        source_train_jsonl=source_train_jsonl,
        source_validation_jsonl=source_validation_jsonl,
        source_test_jsonl=source_test_jsonl,
        labeled_count_per_class=labeled_count_per_class,
        seed=seed,
        source_train_rows=source_train_rows,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        artifacts=artifacts,
    )
    artifacts.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    artifacts.summary_json.write_text(
        json.dumps(
            _build_query_ssl_split_summary(manifest),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return artifacts


def _split_labeled_and_unlabeled_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    labeled_count_per_class: int,
    seed: int,
) -> tuple[list[LabeledQueryRow], list[LabeledQueryRow]]:
    rows_by_label = group_labeled_query_rows_by_label(rows)
    rng = random.Random(seed)

    labeled_rows: list[LabeledQueryRow] = []
    unlabeled_rows: list[LabeledQueryRow] = []
    for label in sorted(rows_by_label):
        bucket = list(rows_by_label[label])
        if len(bucket) < labeled_count_per_class:
            raise ValueError(
                f"Label {label!r} has only {len(bucket)} rows, but "
                f"labeled_count_per_class={labeled_count_per_class} was requested."
            )
        rng.shuffle(bucket)
        labeled_rows.extend(bucket[:labeled_count_per_class])
        unlabeled_rows.extend(bucket[labeled_count_per_class:])

    rng.shuffle(labeled_rows)
    rng.shuffle(unlabeled_rows)
    return labeled_rows, unlabeled_rows


def _build_query_ssl_split_manifest(
    *,
    split_name: str,
    source_train_jsonl: Path,
    source_validation_jsonl: Path,
    source_test_jsonl: Path,
    labeled_count_per_class: int,
    seed: int,
    source_train_rows: Sequence[LabeledQueryRow],
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    validation_rows: Sequence[LabeledQueryRow],
    test_rows: Sequence[LabeledQueryRow],
    artifacts: QuerySslSplitArtifacts,
) -> dict[str, object]:
    return {
        "schema_version": QUERY_SSL_SPLIT_SCHEMA_VERSION,
        "split_name": split_name,
        "source_train_jsonl": str(source_train_jsonl),
        "source_validation_jsonl": str(source_validation_jsonl),
        "source_test_jsonl": str(source_test_jsonl),
        "seed": seed,
        "labeled_policy": {
            "type": "class_balanced_fixed_count_per_class",
            "labeled_count_per_class": labeled_count_per_class,
        },
        "unlabeled_policy": {
            "type": "remaining_train_pool_after_labeled_selection",
            "label_visibility": (
                "retained_in_artifact_for_audit_and_stratified_metrics; "
                "central_ssl_unlabeled_loaders_must_not_consume_labels"
            ),
        },
        "row_counts": {
            "source_train": len(source_train_rows),
            "labeled_train": len(labeled_rows),
            "unlabeled_pool": len(unlabeled_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "label_counts": {
            "source_train": count_labeled_query_rows_by_label(source_train_rows),
            "labeled_train": count_labeled_query_rows_by_label(labeled_rows),
            "unlabeled_pool": count_labeled_query_rows_by_label(unlabeled_rows),
            "validation": count_labeled_query_rows_by_label(validation_rows),
            "test": count_labeled_query_rows_by_label(test_rows),
        },
        "artifacts": {
            "labeled_train_jsonl": str(artifacts.labeled_train_jsonl),
            "unlabeled_pool_jsonl": str(artifacts.unlabeled_pool_jsonl),
            "validation_jsonl": str(artifacts.validation_jsonl),
            "test_jsonl": str(artifacts.test_jsonl),
            "manifest_json": str(artifacts.manifest_json),
            "summary_json": str(artifacts.summary_json),
        },
    }


def _build_query_ssl_split_summary(
    manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": manifest["schema_version"],
        "split_name": manifest["split_name"],
        "seed": manifest["seed"],
        "labeled_policy": manifest["labeled_policy"],
        "unlabeled_policy": manifest["unlabeled_policy"],
        "row_counts": manifest["row_counts"],
        "label_counts": manifest["label_counts"],
    }
