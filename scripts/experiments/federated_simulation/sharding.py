"""Federated simulation shard 분할 로직."""

from __future__ import annotations

import random
from collections import defaultdict

from scripts.experiments.federated_simulation.models import (
    FederatedClientShard,
    FederatedDatasetSplit,
    FederatedShardPolicyConfig,
)
from scripts.labeled_query_rows import LabeledQueryRow


def split_rows_for_federation(
    rows: list[LabeledQueryRow],
    *,
    bootstrap_ratio: float,
    client_count: int,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
) -> FederatedDatasetSplit:
    """train row를 prototype bootstrap과 non-IID client shard로 나눈다."""
    if not 0.0 < bootstrap_ratio < 1.0:
        raise ValueError("bootstrap_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    rows_by_label = _group_shuffled_rows_by_label(rows, rng)
    bootstrap_rows: list[LabeledQueryRow] = []
    remaining_by_label: dict[str, list[LabeledQueryRow]] = {}
    for label in sorted(rows_by_label):
        bucket = list(rows_by_label[label])
        bootstrap_count = int(round(len(bucket) * bootstrap_ratio))
        if bootstrap_count <= 0 and len(bucket) > 1:
            bootstrap_count = 1
        if bootstrap_count >= len(bucket):
            bootstrap_count = len(bucket) - 1
        bootstrap_rows.extend(bucket[:bootstrap_count])
        remaining_by_label[label] = bucket[bootstrap_count:]

    client_shards = _assign_client_shards(
        remaining_by_label=remaining_by_label,
        client_count=client_count,
        shard_policy=shard_policy,
        rng=rng,
    )

    return FederatedDatasetSplit(
        bootstrap_rows=bootstrap_rows,
        client_shards=client_shards,
    )


def split_rows_into_client_shards(
    rows: list[LabeledQueryRow],
    *,
    client_count: int,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
) -> tuple[FederatedClientShard, ...]:
    """heldout validation row를 bootstrap 없이 client shard로 나눈다."""
    rng = random.Random(seed)
    return _assign_client_shards(
        remaining_by_label=_group_shuffled_rows_by_label(rows, rng),
        client_count=client_count,
        shard_policy=shard_policy,
        rng=rng,
    )


def _group_shuffled_rows_by_label(
    rows: list[LabeledQueryRow],
    rng: random.Random,
) -> dict[str, list[LabeledQueryRow]]:
    rows_by_label: dict[str, list[LabeledQueryRow]] = defaultdict(list)
    for row in rows:
        rows_by_label[str(row["mapped_label_4"])].append(row)
    for label in sorted(rows_by_label):
        rng.shuffle(rows_by_label[label])
    return rows_by_label


def _assign_client_shards(
    *,
    remaining_by_label: dict[str, list[LabeledQueryRow]],
    client_count: int,
    shard_policy: FederatedShardPolicyConfig,
    rng: random.Random,
) -> tuple[FederatedClientShard, ...]:
    if client_count <= 0:
        raise ValueError("client_count must be positive.")

    client_shards = [
        FederatedClientShard(
            client_id=f"{shard_policy.client_id_prefix}_{index + 1:02d}",
            rows=[],
        )
        for index in range(client_count)
    ]

    if shard_policy.name == "label_dominant":
        _assign_label_dominant_shards(
            client_shards=client_shards,
            remaining_by_label=remaining_by_label,
            shard_policy=shard_policy,
        )
    elif shard_policy.name == "dirichlet_label_skew":
        _assign_dirichlet_label_skew_shards(
            client_shards=client_shards,
            remaining_by_label=remaining_by_label,
            shard_policy=shard_policy,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported federated shard policy: {shard_policy.name}")

    for shard in client_shards:
        rng.shuffle(shard.rows)
    return tuple(client_shards)


def _assign_label_dominant_shards(
    *,
    client_shards: list[FederatedClientShard],
    remaining_by_label: dict[str, list[LabeledQueryRow]],
    shard_policy: FederatedShardPolicyConfig,
) -> None:
    dominant_ratio = shard_policy.dominant_ratio
    if dominant_ratio is None or not 0.0 < dominant_ratio <= 1.0:
        raise ValueError("shard_policy.dominant_ratio must be between 0 and 1.")

    client_count = len(client_shards)
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
        client_shards[dominant_index].rows.extend(bucket[:dominant_count])
        if secondary_index != dominant_index:
            client_shards[secondary_index].rows.extend(bucket[dominant_count:])


def _assign_dirichlet_label_skew_shards(
    *,
    client_shards: list[FederatedClientShard],
    remaining_by_label: dict[str, list[LabeledQueryRow]],
    shard_policy: FederatedShardPolicyConfig,
    rng: random.Random,
) -> None:
    alpha = shard_policy.alpha
    if alpha is None or alpha <= 0.0:
        raise ValueError("shard_policy.alpha must be positive.")

    client_count = len(client_shards)
    for label in sorted(remaining_by_label):
        bucket = list(remaining_by_label[label])
        counts = _sample_dirichlet_counts(
            total=len(bucket),
            parts=client_count,
            alpha=alpha,
            rng=rng,
        )
        offset = 0
        for shard, count in zip(client_shards, counts, strict=True):
            if count <= 0:
                continue
            shard.rows.extend(bucket[offset : offset + count])
            offset += count


def _sample_dirichlet_counts(
    *,
    total: int,
    parts: int,
    alpha: float,
    rng: random.Random,
) -> list[int]:
    if total <= 0:
        return [0 for _ in range(parts)]

    weights = [rng.gammavariate(alpha, 1.0) for _ in range(parts)]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return [0 for _ in range(parts - 1)] + [total]

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
    return counts
