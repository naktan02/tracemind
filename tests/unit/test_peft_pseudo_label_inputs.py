from __future__ import annotations

import pytest

from methods.ssl.pseudo_label_replay import build_pseudo_label_replay_rows
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


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


def test_build_pseudo_label_replay_rows_combines_seed_and_pseudo_rows() -> None:
    resolved = build_pseudo_label_replay_rows(
        seed_train_rows=[_row("seed_q1", "anxiety", raw_label_scheme="manual_label")],
        pseudo_label_rows=[_row("pseudo_q1", "depression")],
    )

    assert [row["query_id"] for row in resolved.seed_train_rows] == ["seed_q1"]
    assert [row["query_id"] for row in resolved.pseudo_label_rows] == ["pseudo_q1"]
    assert [row["query_id"] for row in resolved.combined_train_rows] == [
        "seed_q1",
        "pseudo_q1",
    ]


def test_build_pseudo_label_replay_rows_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError, match="pseudo_label_rows"):
        build_pseudo_label_replay_rows(
            seed_train_rows=[],
            pseudo_label_rows=[
                _row("pseudo_q1", "depression"),
                _row("pseudo_q1", "depression"),
            ],
        )


def test_build_pseudo_label_replay_rows_rejects_seed_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint query_id"):
        build_pseudo_label_replay_rows(
            seed_train_rows=[
                _row("same_q", "anxiety", raw_label_scheme="manual_label")
            ],
            pseudo_label_rows=[_row("same_q", "depression")],
        )
