"""Pseudo-label self-training 입력 row 정규화 helper."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from scripts.experiments.query_lora_ssl.io.query_adaptation import (
    build_labeled_rows_from_query_adaptation_dataset,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


@dataclass(slots=True)
class ResolvedPseudoLabelTrainingRows:
    """self-training에 사용할 seed/pseudo/combined rows."""

    seed_train_rows: list[LabeledQueryRow]
    pseudo_label_rows: list[LabeledQueryRow]
    combined_train_rows: list[LabeledQueryRow]


def resolve_pseudo_label_training_rows(
    *,
    cfg: DictConfig,
    pseudo_label_jsonl: str | Path | None,
    pseudo_label_rows: Sequence[LabeledQueryRow] | None,
    pseudo_label_dataset: Any | None,
    seed_train_rows: Sequence[LabeledQueryRow] | None,
    include_seed_train_rows: bool | None,
    train_jsonl_ref: str | Path | None,
) -> ResolvedPseudoLabelTrainingRows:
    """self-training 입력 row source를 하나의 combined row list로 정규화한다."""

    effective_include_seed_rows = (
        bool(getattr(cfg, "include_seed_train_rows", True))
        if include_seed_train_rows is None
        else bool(include_seed_train_rows)
    )
    if effective_include_seed_rows:
        effective_seed_train_rows = (
            load_labeled_query_rows(
                Path(
                    str(cfg.train_jsonl if train_jsonl_ref is None else train_jsonl_ref)
                )
            )
            if seed_train_rows is None
            else list(seed_train_rows)
        )
    else:
        effective_seed_train_rows = []

    effective_pseudo_label_rows = _resolve_pseudo_label_rows(
        cfg=cfg,
        pseudo_label_jsonl=pseudo_label_jsonl,
        pseudo_label_rows=pseudo_label_rows,
        pseudo_label_dataset=pseudo_label_dataset,
    )
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


def _resolve_pseudo_label_rows(
    *,
    cfg: DictConfig,
    pseudo_label_jsonl: str | Path | None,
    pseudo_label_rows: Sequence[LabeledQueryRow] | None,
    pseudo_label_dataset: Any | None,
) -> list[LabeledQueryRow]:
    provided_sources = sum(
        source is not None
        for source in (pseudo_label_jsonl, pseudo_label_rows, pseudo_label_dataset)
    )
    if provided_sources > 1:
        raise ValueError(
            "Provide only one of pseudo_label_jsonl, pseudo_label_rows, or "
            "pseudo_label_dataset."
        )
    if pseudo_label_dataset is not None:
        return build_labeled_rows_from_query_adaptation_dataset(
            pseudo_label_dataset,
            annotation_source="pseudo_label_self_training",
        )
    if pseudo_label_rows is not None:
        return list(pseudo_label_rows)

    effective_path = pseudo_label_jsonl or getattr(cfg, "pseudo_label_jsonl", None)
    if effective_path is None:
        raise ValueError(
            "pseudo_label_jsonl is required when pseudo_label_rows/dataset is not "
            "provided."
        )
    return load_labeled_query_rows(Path(str(effective_path)))


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
