from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.datasets.lib.query_ssl_views import (
    QUERY_SSL_VIEWS_SCHEMA_VERSION,
    materialize_query_ssl_backtranslation_views,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)


@dataclass(frozen=True, slots=True)
class _FakePair:
    aug_0: str
    aug_1: str
    aug_0_pivot_lang: str = "deu_Latn"
    aug_1_pivot_lang: str = "fra_Latn"


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


def _fake_builder(texts):
    return [_FakePair(aug_0=f"{text} aug0", aug_1=f"{text} aug1") for text in texts]


def test_materialize_query_ssl_views_writes_labeled_and_unlabeled_views(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    dump_labeled_query_rows(
        split_dir / "labeled_train.jsonl",
        [_row("l1", "anxiety"), _row("l2", "depression")],
    )
    dump_labeled_query_rows(
        split_dir / "unlabeled_pool.jsonl",
        [_row("u1", "normal")],
    )
    (split_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    artifacts = materialize_query_ssl_backtranslation_views(
        split_dir=split_dir,
        split_name="unit_split",
        augmenter_name="fake_backtranslation",
        output_root=tmp_path / "views",
        augmenter_manifest={"augmenter_type": "fake"},
        candidate_pair_builder=_fake_builder,
    )

    labeled_rows = load_labeled_query_rows(artifacts.labeled_train_with_views_jsonl)
    unlabeled_rows = load_labeled_query_rows(artifacts.unlabeled_pool_with_views_jsonl)
    assert labeled_rows[0]["aug_0"].endswith("aug0")
    assert labeled_rows[0]["aug_1"].endswith("aug1")
    assert unlabeled_rows[0]["aug_0_pivot_lang"] == "deu_Latn"
    assert unlabeled_rows[0]["aug_1_pivot_lang"] == "fra_Latn"

    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == QUERY_SSL_VIEWS_SCHEMA_VERSION
    assert manifest["row_counts"] == {
        "labeled_train": 2,
        "unlabeled_pool": 1,
    }
    assert manifest["view_counts"]["labeled_train"]["empty_aug_0_count"] == 0
    assert manifest["view_counts"]["unlabeled_pool"]["empty_aug_1_count"] == 0


def test_materialize_query_ssl_views_rejects_empty_candidate_text(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    dump_labeled_query_rows(split_dir / "labeled_train.jsonl", [_row("l1", "anxiety")])
    dump_labeled_query_rows(split_dir / "unlabeled_pool.jsonl", [])

    def _bad_builder(texts):
        return [_FakePair(aug_0="", aug_1="strong") for _ in texts]

    with pytest.raises(ValueError, match="non-empty aug_0 and aug_1"):
        materialize_query_ssl_backtranslation_views(
            split_dir=split_dir,
            split_name="unit_split",
            augmenter_name="fake_backtranslation",
            output_root=tmp_path / "views",
            augmenter_manifest={"augmenter_type": "fake"},
            candidate_pair_builder=_bad_builder,
        )
