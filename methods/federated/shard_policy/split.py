"""Federated shard policy split algorithms."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TypeVar

from methods.federated.shard_policy.base import (
    SHARD_POLICY_DIRICHLET_LABEL_SKEW,
    SHARD_POLICY_LABEL_DOMINANT,
    FederatedClientShardAssignment,
    FederatedShardPolicyConfig,
    FederatedShardSplit,
)

ItemT = TypeVar("ItemT")


def split_items_for_federation(
    items: Sequence[ItemT],
    *,
    label_getter: Callable[[ItemT], str],
    bootstrap_ratio: float,
    client_count: int,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
) -> FederatedShardSplit[ItemT]:
    """item을 bootstrap set과 non-IID client shard로 나눈다."""
    if not 0.0 < bootstrap_ratio < 1.0:
        raise ValueError("bootstrap_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    items_by_label = _group_shuffled_items_by_label(items, label_getter, rng)
    bootstrap_items: list[ItemT] = []
    remaining_by_label: dict[str, list[ItemT]] = {}
    for label in sorted(items_by_label):
        bucket = list(items_by_label[label])
        bootstrap_count = int(round(len(bucket) * bootstrap_ratio))
        if bootstrap_count <= 0 and len(bucket) > 1:
            bootstrap_count = 1
        if bootstrap_count >= len(bucket):
            bootstrap_count = len(bucket) - 1
        bootstrap_items.extend(bucket[:bootstrap_count])
        remaining_by_label[label] = bucket[bootstrap_count:]

    return FederatedShardSplit(
        bootstrap_items=tuple(bootstrap_items),
        client_shards=assign_items_to_client_shards(
            remaining_by_label=remaining_by_label,
            client_count=client_count,
            shard_policy=shard_policy,
            rng=rng,
        ),
    )


def split_items_into_client_shards(
    items: Sequence[ItemT],
    *,
    label_getter: Callable[[ItemT], str],
    client_count: int,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
) -> tuple[FederatedClientShardAssignment[ItemT], ...]:
    """item을 bootstrap 없이 client shard로 나눈다."""
    rng = random.Random(seed)
    return assign_items_to_client_shards(
        remaining_by_label=_group_shuffled_items_by_label(items, label_getter, rng),
        client_count=client_count,
        shard_policy=shard_policy,
        rng=rng,
    )


def assign_items_to_client_shards(
    *,
    remaining_by_label: dict[str, list[ItemT]],
    client_count: int,
    shard_policy: FederatedShardPolicyConfig,
    rng: random.Random,
) -> tuple[FederatedClientShardAssignment[ItemT], ...]:
    """label별 item bucket을 shard policy에 따라 client별로 배정한다."""
    if client_count <= 0:
        raise ValueError("client_count must be positive.")

    client_items: list[list[ItemT]] = [[] for _ in range(client_count)]
    if shard_policy.name == SHARD_POLICY_LABEL_DOMINANT:
        _assign_label_dominant_shards(
            client_items=client_items,
            remaining_by_label=remaining_by_label,
            shard_policy=shard_policy,
        )
    elif shard_policy.name == SHARD_POLICY_DIRICHLET_LABEL_SKEW:
        _assign_dirichlet_label_skew_shards(
            client_items=client_items,
            remaining_by_label=remaining_by_label,
            shard_policy=shard_policy,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported federated shard policy: {shard_policy.name}")

    assignments: list[FederatedClientShardAssignment[ItemT]] = []
    for index, items in enumerate(client_items):
        rng.shuffle(items)
        assignments.append(
            FederatedClientShardAssignment(
                client_id=f"{shard_policy.client_id_prefix}_{index + 1:02d}",
                items=tuple(items),
            )
        )
    return tuple(assignments)


def sample_dirichlet_counts(
    *,
    total: int,
    parts: int,
    alpha: float,
    rng: random.Random,
) -> tuple[int, ...]:
    """Dirichlet 비율을 integer assignment count로 변환한다."""
    if total <= 0:
        return tuple(0 for _ in range(parts))

    weights = [rng.gammavariate(alpha, 1.0) for _ in range(parts)]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return tuple([0 for _ in range(parts - 1)] + [total])

    raw_counts = [total * weight / weight_sum for weight in weights]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    fractional_order = sorted(
        range(parts),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _group_shuffled_items_by_label(
    items: Sequence[ItemT],
    label_getter: Callable[[ItemT], str],
    rng: random.Random,
) -> dict[str, list[ItemT]]:
    items_by_label: dict[str, list[ItemT]] = defaultdict(list)
    for item in items:
        items_by_label[label_getter(item)].append(item)
    for label in sorted(items_by_label):
        rng.shuffle(items_by_label[label])
    return items_by_label


def _assign_label_dominant_shards(
    *,
    client_items: list[list[ItemT]],
    remaining_by_label: dict[str, list[ItemT]],
    shard_policy: FederatedShardPolicyConfig,
) -> None:
    dominant_ratio = shard_policy.dominant_ratio
    if dominant_ratio is None or not 0.0 < dominant_ratio <= 1.0:
        raise ValueError("shard_policy.dominant_ratio must be between 0 and 1.")

    client_count = len(client_items)
    labels = sorted(remaining_by_label)
    for label_index, label in enumerate(labels):
        bucket = list(remaining_by_label[label])
        dominant_index = label_index % client_count
        secondary_index = (
            (label_index + 1) % client_count if client_count > 1 else dominant_index
        )
        dominant_count = int(round(len(bucket) * dominant_ratio))
        dominant_count = max(0, min(len(bucket), dominant_count))
        if dominant_count == 0 and bucket:
            dominant_count = len(bucket)
        client_items[dominant_index].extend(bucket[:dominant_count])
        if secondary_index != dominant_index:
            client_items[secondary_index].extend(bucket[dominant_count:])


def _assign_dirichlet_label_skew_shards(
    *,
    client_items: list[list[ItemT]],
    remaining_by_label: dict[str, list[ItemT]],
    shard_policy: FederatedShardPolicyConfig,
    rng: random.Random,
) -> None:
    alpha = shard_policy.alpha
    if alpha is None or alpha <= 0.0:
        raise ValueError("shard_policy.alpha must be positive.")

    client_count = len(client_items)
    for label in sorted(remaining_by_label):
        bucket = list(remaining_by_label[label])
        counts = sample_dirichlet_counts(
            total=len(bucket),
            parts=client_count,
            alpha=alpha,
            rng=rng,
        )
        offset = 0
        for items, count in zip(client_items, counts, strict=True):
            if count <= 0:
                continue
            items.extend(bucket[offset : offset + count])
            offset += count
