from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.datasets.lib.query_ssl_split import (
    QUERY_SSL_SPLIT_SCHEMA_VERSION,
    materialize_class_balanced_query_ssl_split,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)


def _row(query_id: str, label: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=f"text {query_id}",
        raw_label_scheme="manual_label",
        raw_label=label,
        mapped_label_4=label,
        locale="eng_Latn",
        annotation_source="unit_test",
        approved_by="tester",
        created_at="2026-05-10T00:00:00+00:00",
    )


def test_materialize_query_ssl_split_balances_labeled_and_keeps_remaining_unlabeled(
    tmp_path: Path,
) -> None:
    source_train = tmp_path / "train_pool.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    dump_labeled_query_rows(
        source_train,
        [
            _row("a1", "anxiety"),
            _row("a2", "anxiety"),
            _row("a3", "anxiety"),
            _row("d1", "depression"),
            _row("d2", "depression"),
            _row("d3", "depression"),
        ],
    )
    dump_labeled_query_rows(validation, [_row("v1", "anxiety")])
    dump_labeled_query_rows(test, [_row("t1", "depression")])

    artifacts = materialize_class_balanced_query_ssl_split(
        source_train_jsonl=source_train,
        source_validation_jsonl=validation,
        source_test_jsonl=test,
        split_name="unit_ssl_split",
        labeled_count_per_class=2,
        seed=42,
        output_root=tmp_path / "out",
    )

    labeled_rows = load_labeled_query_rows(artifacts.labeled_train_jsonl)
    unlabeled_rows = load_labeled_query_rows(artifacts.unlabeled_pool_jsonl)
    labeled_ids = {row["query_id"] for row in labeled_rows}
    unlabeled_ids = {row["query_id"] for row in unlabeled_rows}

    assert len(labeled_rows) == 4
    assert len(unlabeled_rows) == 2
    assert labeled_ids.isdisjoint(unlabeled_ids)
    assert labeled_ids | unlabeled_ids == {"a1", "a2", "a3", "d1", "d2", "d3"}
    assert load_labeled_query_rows(artifacts.validation_jsonl)[0]["query_id"] == "v1"
    assert load_labeled_query_rows(artifacts.test_jsonl)[0]["query_id"] == "t1"

    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == QUERY_SSL_SPLIT_SCHEMA_VERSION
    assert manifest["row_counts"]["labeled_train"] == 4
    assert manifest["row_counts"]["unlabeled_pool"] == 2
    assert manifest["label_counts"]["labeled_train"] == {
        "anxiety": 2,
        "depression": 2,
    }
    assert manifest["label_counts"]["unlabeled_pool"] == {
        "anxiety": 1,
        "depression": 1,
    }


def test_materialize_query_ssl_split_rejects_insufficient_class_budget(
    tmp_path: Path,
) -> None:
    source_train = tmp_path / "train_pool.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    dump_labeled_query_rows(source_train, [_row("a1", "anxiety")])
    dump_labeled_query_rows(validation, [_row("v1", "anxiety")])
    dump_labeled_query_rows(test, [_row("t1", "anxiety")])

    with pytest.raises(ValueError, match="labeled_count_per_class"):
        materialize_class_balanced_query_ssl_split(
            source_train_jsonl=source_train,
            source_validation_jsonl=validation,
            source_test_jsonl=test,
            split_name="unit_ssl_split",
            labeled_count_per_class=2,
            seed=42,
            output_root=tmp_path / "out",
        )
