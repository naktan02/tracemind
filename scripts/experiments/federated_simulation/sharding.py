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
    if shard_policy.name != "label_dominant":
        raise ValueError(f"Unsupported federated shard policy: {shard_policy.name}")
    if not 0.0 < bootstrap_ratio < 1.0:
        raise ValueError("bootstrap_ratio must be between 0 and 1.")
    if client_count <= 0:
        raise ValueError("client_count must be positive.")
    if not 0.0 < shard_policy.dominant_ratio <= 1.0:
        raise ValueError("shard_policy.dominant_ratio must be between 0 and 1.")

    rows_by_label: dict[str, list[LabeledQueryRow]] = defaultdict(list)
    for row in rows:
        rows_by_label[str(row["mapped_label_4"])].append(row)

    rng = random.Random(seed)
    bootstrap_rows: list[LabeledQueryRow] = []
    remaining_by_label: dict[str, list[LabeledQueryRow]] = {}
    for label in sorted(rows_by_label):
        bucket = list(rows_by_label[label])
        rng.shuffle(bucket)
        bootstrap_count = int(round(len(bucket) * bootstrap_ratio))
        if bootstrap_count <= 0 and len(bucket) > 1:
            bootstrap_count = 1
        if bootstrap_count >= len(bucket):
            bootstrap_count = len(bucket) - 1
        bootstrap_rows.extend(bucket[:bootstrap_count])
        remaining_by_label[label] = bucket[bootstrap_count:]

    client_shards = [
        FederatedClientShard(
            client_id=f"{shard_policy.client_id_prefix}_{index + 1:02d}",
            rows=[],
        )
        for index in range(client_count)
    ]
    labels = sorted(remaining_by_label)
    for label_index, label in enumerate(labels):
        bucket = list(remaining_by_label[label])
        dominant_index = label_index % client_count
        secondary_index = (
            (label_index + 1) % client_count if client_count > 1 else dominant_index
        )
        dominant_count = int(round(len(bucket) * shard_policy.dominant_ratio))
        dominant_count = max(0, min(len(bucket), dominant_count))
        if dominant_count == 0 and bucket:
            dominant_count = len(bucket)
        client_shards[dominant_index].rows.extend(bucket[:dominant_count])
        if secondary_index != dominant_index:
            client_shards[secondary_index].rows.extend(bucket[dominant_count:])

    for shard in client_shards:
        rng.shuffle(shard.rows)

    return FederatedDatasetSplit(
        bootstrap_rows=bootstrap_rows,
        client_shards=tuple(client_shards),
    )
