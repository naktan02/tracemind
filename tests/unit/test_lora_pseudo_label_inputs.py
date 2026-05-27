from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.experiments.query_peft_ssl.runners.pseudo_label_inputs import (
    resolve_pseudo_label_training_rows,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)


def _cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "train_jsonl": str(tmp_path / "seed_train.jsonl"),
            "pseudo_label_jsonl": None,
            "include_seed_train_rows": True,
        }
    )


def _row(
    query_id: str,
    label: str,
    *,
    raw_label_scheme: str = "pseudo_label",
) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=f"text for {query_id}",
        raw_label_scheme=raw_label_scheme,
        raw_label=label,
        mapped_label_4=label,
        locale="ko-KR",
        annotation_source=raw_label_scheme,
        approved_by=None,
        created_at="2026-04-12T12:00:00+00:00",
    )


def test_resolve_pseudo_label_training_rows_combines_seed_and_pseudo_jsonl(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    pseudo_label_path = tmp_path / "pseudo_label.jsonl"
    dump_labeled_query_rows(
        Path(str(cfg.train_jsonl)),
        [_row("seed_q1", "anxiety", raw_label_scheme="manual_label")],
    )
    dump_labeled_query_rows(
        pseudo_label_path,
        [_row("pseudo_q1", "depression")],
    )

    resolved = resolve_pseudo_label_training_rows(
        cfg=cfg,
        pseudo_label_jsonl=pseudo_label_path,
        pseudo_label_rows=None,
        pseudo_label_dataset=None,
        seed_train_rows=None,
        include_seed_train_rows=None,
        train_jsonl_ref=None,
    )

    assert [row["query_id"] for row in resolved.seed_train_rows] == ["seed_q1"]
    assert [row["query_id"] for row in resolved.pseudo_label_rows] == ["pseudo_q1"]
    assert [row["query_id"] for row in resolved.combined_train_rows] == [
        "seed_q1",
        "pseudo_q1",
    ]


def test_resolve_pseudo_label_training_rows_rejects_multiple_sources(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)

    with pytest.raises(ValueError, match="Provide only one"):
        resolve_pseudo_label_training_rows(
            cfg=cfg,
            pseudo_label_jsonl=tmp_path / "pseudo_label.jsonl",
            pseudo_label_rows=[_row("pseudo_q1", "depression")],
            pseudo_label_dataset=None,
            seed_train_rows=[],
            include_seed_train_rows=False,
            train_jsonl_ref=None,
        )


def test_resolve_pseudo_label_training_rows_rejects_duplicate_query_ids(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)

    with pytest.raises(ValueError, match="pseudo_label_rows"):
        resolve_pseudo_label_training_rows(
            cfg=cfg,
            pseudo_label_jsonl=None,
            pseudo_label_rows=[
                _row("pseudo_q1", "depression"),
                _row("pseudo_q1", "depression"),
            ],
            pseudo_label_dataset=None,
            seed_train_rows=[],
            include_seed_train_rows=False,
            train_jsonl_ref=None,
        )


def test_resolve_pseudo_label_training_rows_rejects_seed_overlap(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)

    with pytest.raises(ValueError, match="disjoint query_id"):
        resolve_pseudo_label_training_rows(
            cfg=cfg,
            pseudo_label_jsonl=None,
            pseudo_label_rows=[_row("same_q", "depression")],
            pseudo_label_dataset=None,
            seed_train_rows=[
                _row("same_q", "anxiety", raw_label_scheme="manual_label")
            ],
            include_seed_train_rows=True,
            train_jsonl_ref=None,
        )
