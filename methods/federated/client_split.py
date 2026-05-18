"""Federated client split policy helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class FederatedLabeledPoolPolicy:
    """FL materialized split에 포함할 labeled source pool 선택 정책."""

    mode: str = "all"
    count_per_class: int | None = None
    fraction: float | None = None

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
    ) -> "FederatedLabeledPoolPolicy":
        raw_count = source.get("count_per_class")
        raw_fraction = source.get("fraction")
        return cls(
            mode=str(source.get("mode", "all")),
            count_per_class=None if raw_count is None else int(raw_count),
            fraction=None if raw_fraction is None else float(raw_fraction),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "count_per_class": self.count_per_class,
            "fraction": self.fraction,
        }


def select_labeled_pool_items(
    items: Sequence[ItemT],
    *,
    policy: FederatedLabeledPoolPolicy,
    seed: int,
    label_getter: Callable[[ItemT], str],
) -> list[ItemT]:
    """source labeled pool에서 materialized FL split에 쓸 item을 선택한다."""

    if policy.mode == "all":
        if policy.count_per_class is not None or policy.fraction is not None:
            raise ValueError(
                "labeled_policy.count_per_class and labeled_policy.fraction must be "
                "null when mode is all."
            )
        return list(items)
    if policy.mode == "count_per_class":
        if policy.fraction is not None:
            raise ValueError(
                "labeled_policy.fraction must be null when mode is count_per_class."
            )
        return _select_items_by_count_per_class(
            items,
            count_per_class=policy.count_per_class,
            seed=seed,
            label_getter=label_getter,
        )
    if policy.mode == "fraction":
        if policy.count_per_class is not None:
            raise ValueError(
                "labeled_policy.count_per_class must be null when mode is fraction."
            )
        return _select_items_by_fraction(
            items,
            fraction=policy.fraction,
            seed=seed,
            label_getter=label_getter,
        )
    raise ValueError(
        "fl_client_split_materialization.labeled_policy.mode must be one of "
        "'all', 'count_per_class', or 'fraction'."
    )


def split_client_pool_items(
    items: Sequence[ItemT],
    *,
    labeled_ratio: float,
    seed: int,
    label_getter: Callable[[ItemT], str],
) -> tuple[list[ItemT], list[ItemT]]:
    """client shard 내부 rows를 class별 labeled/unlabeled pool로 나눈다."""

    rng = random.Random(seed)
    labeled_items: list[ItemT] = []
    unlabeled_items: list[ItemT] = []
    items_by_label = _group_items_by_label(items, label_getter=label_getter)

    for label in sorted(items_by_label):
        bucket = list(items_by_label[label])
        rng.shuffle(bucket)
        labeled_count = _resolve_labeled_count(
            bucket_size=len(bucket),
            labeled_ratio=labeled_ratio,
        )
        labeled_items.extend(bucket[:labeled_count])
        unlabeled_items.extend(bucket[labeled_count:])
    return labeled_items, unlabeled_items


def _select_items_by_count_per_class(
    items: Sequence[ItemT],
    *,
    count_per_class: int | None,
    seed: int,
    label_getter: Callable[[ItemT], str],
) -> list[ItemT]:
    if count_per_class is None or count_per_class <= 0:
        raise ValueError(
            "labeled_policy.count_per_class must be positive when mode is "
            "count_per_class."
        )
    rng = random.Random(seed)
    selected_items: list[ItemT] = []
    buckets = _group_items_by_label(items, label_getter=label_getter)
    for label, bucket in buckets.items():
        if len(bucket) < count_per_class:
            raise ValueError(
                "labeled_policy.count_per_class exceeds source labeled rows for "
                f"{label}: {count_per_class} > {len(bucket)}."
            )
        shuffled_bucket = list(bucket)
        rng.shuffle(shuffled_bucket)
        selected_items.extend(shuffled_bucket[:count_per_class])
    rng.shuffle(selected_items)
    return selected_items


def _select_items_by_fraction(
    items: Sequence[ItemT],
    *,
    fraction: float | None,
    seed: int,
    label_getter: Callable[[ItemT], str],
) -> list[ItemT]:
    if fraction is None or not 0.0 < fraction <= 1.0:
        raise ValueError(
            "labeled_policy.fraction must be between 0 and 1 when mode is fraction."
        )
    rng = random.Random(seed)
    selected_items: list[ItemT] = []
    for bucket in _group_items_by_label(items, label_getter=label_getter).values():
        selected_count = int(round(len(bucket) * fraction))
        if selected_count <= 0 and bucket:
            selected_count = 1
        selected_count = min(selected_count, len(bucket))
        shuffled_bucket = list(bucket)
        rng.shuffle(shuffled_bucket)
        selected_items.extend(shuffled_bucket[:selected_count])
    rng.shuffle(selected_items)
    return selected_items


def _resolve_labeled_count(*, bucket_size: int, labeled_ratio: float) -> int:
    if bucket_size <= 0 or labeled_ratio <= 0.0:
        return 0
    if labeled_ratio >= 1.0:
        return bucket_size
    labeled_count = int(round(bucket_size * labeled_ratio))
    if labeled_count <= 0 and bucket_size > 1:
        labeled_count = 1
    if labeled_count >= bucket_size:
        labeled_count = bucket_size - 1
    return labeled_count


def _group_items_by_label(
    items: Sequence[ItemT],
    *,
    label_getter: Callable[[ItemT], str],
) -> dict[str, list[ItemT]]:
    buckets: dict[str, list[ItemT]] = defaultdict(list)
    for item in items:
        buckets[label_getter(item)].append(item)
    return buckets
