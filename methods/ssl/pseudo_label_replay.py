"""Pseudo-label replay 학습 row 의미를 정규화한다."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(slots=True)
class ResolvedPseudoLabelTrainingRows:
    """self-training에 사용할 seed/pseudo/combined rows."""

    seed_train_rows: list[LabeledQueryRow]
    pseudo_label_rows: list[LabeledQueryRow]
    combined_train_rows: list[LabeledQueryRow]


def build_pseudo_label_replay_rows(
    *,
    seed_train_rows: Sequence[LabeledQueryRow] | None,
    pseudo_label_rows: Sequence[LabeledQueryRow],
) -> ResolvedPseudoLabelTrainingRows:
    """self-training 입력 row를 하나의 combined row list로 정규화한다."""

    effective_seed_train_rows = [] if seed_train_rows is None else list(seed_train_rows)
    effective_pseudo_label_rows = list(pseudo_label_rows)
    if not effective_pseudo_label_rows:
        raise ValueError("pseudo_label_rows must not be empty.")

    _ensure_unique_query_ids(
        effective_seed_train_rows,
        item_name="seed_train_rows",
    )
    _ensure_unique_query_ids(
        effective_pseudo_label_rows,
        item_name="pseudo_label_rows",
    )
    _ensure_disjoint_query_ids(
        seed_train_rows=effective_seed_train_rows,
        pseudo_label_rows=effective_pseudo_label_rows,
    )

    return ResolvedPseudoLabelTrainingRows(
        seed_train_rows=effective_seed_train_rows,
        pseudo_label_rows=effective_pseudo_label_rows,
        combined_train_rows=[
            *effective_seed_train_rows,
            *effective_pseudo_label_rows,
        ],
    )


def _ensure_unique_query_ids(
    rows: Sequence[LabeledQueryRow],
    *,
    item_name: str,
) -> None:
    counts = Counter(str(row["query_id"]) for row in rows)
    duplicates = sorted(query_id for query_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{item_name} must not contain duplicate query_id values: {duplicates[:5]}."
        )


def _ensure_disjoint_query_ids(
    *,
    seed_train_rows: Sequence[LabeledQueryRow],
    pseudo_label_rows: Sequence[LabeledQueryRow],
) -> None:
    pseudo_query_ids = {str(item["query_id"]) for item in pseudo_label_rows}
    overlapping_query_ids = sorted(
        str(row["query_id"])
        for row in seed_train_rows
        if str(row["query_id"]) in pseudo_query_ids
    )
    if overlapping_query_ids:
        raise ValueError(
            "seed_train_rows and pseudo_label_rows must use disjoint query_id values. "
            f"Found overlaps: {overlapping_query_ids[:5]}."
        )
