"""FL SSL diagnostic/probe row sampling helpers."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    group_labeled_query_rows_by_label,
)

DIAGNOSTIC_VIEW_DETERMINISTIC_RANDOM = "deterministic_random"
PEER_PROBE_LABEL_BALANCED = "label_balanced"


def normalize_sampling_policy_name(policy_name: str) -> str:
    """sampling policy 이름을 canonical 비교용으로 정규화한다."""

    return policy_name.strip().lower().replace("-", "_")


def select_deterministic_diagnostic_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    max_rows: int,
    run_seed: int,
    seed_offset: int,
    round_index: int,
    client_id: str,
) -> tuple[LabeledQueryRow, ...]:
    """client-local diagnostic row subset을 deterministic하게 고른다."""

    if len(rows) <= max_rows:
        return tuple(rows)
    sorted_rows = sorted(rows, key=lambda row: str(row["query_id"]))
    rng = random.Random(
        f"{int(run_seed) + int(seed_offset)}:{int(round_index)}:{client_id}"
    )
    selected_indices = sorted(rng.sample(range(len(sorted_rows)), max_rows))
    return tuple(sorted_rows[index] for index in selected_indices)


def select_label_balanced_probe_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    max_rows: int,
    seed: int,
) -> tuple[LabeledQueryRow, ...]:
    """label-balanced fixed probe subset을 deterministic하게 고른다."""

    rows_by_label = group_labeled_query_rows_by_label(rows)
    rng = random.Random(seed)
    for label in sorted(rows_by_label):
        label_rows = rows_by_label[label]
        label_rows.sort(key=lambda row: str(row["query_id"]))
        rng.shuffle(label_rows)

    selected: list[LabeledQueryRow] = []
    labels = sorted(rows_by_label)
    while len(selected) < max_rows and labels:
        next_labels: list[str] = []
        for label in labels:
            label_rows = rows_by_label[label]
            if not label_rows:
                continue
            selected.append(label_rows.pop(0))
            if len(selected) >= max_rows:
                break
            if label_rows:
                next_labels.append(label)
        labels = next_labels
    return tuple(selected)


def hash_query_ids(query_ids: Sequence[str]) -> str:
    """query id sequence를 report 비교용 digest로 만든다."""

    payload = json.dumps(list(query_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
