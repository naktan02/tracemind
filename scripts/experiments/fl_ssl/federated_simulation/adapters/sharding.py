"""Federated simulation row shard adapter."""

from __future__ import annotations

import random
from collections import defaultdict

from methods.federated.shard_policy.base import (
    FederatedClientShardAssignment,
    FederatedShardPolicyConfig,
)
from methods.federated.shard_policy.split import (
    split_items_for_federation,
    split_items_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedClientShard,
    FederatedDatasetSplit,
)
from scripts.io.labeled_query_rows import LabeledQueryRow


def split_rows_for_federation(
    rows: list[LabeledQueryRow],
    *,
    bootstrap_ratio: float,
    client_count: int,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
    client_pool_split_config: FederatedClientPoolSplitConfig | None = None,
) -> FederatedDatasetSplit:
    """train row를 prototype bootstrap과 non-IID client shard로 나눈다."""
    split = split_items_for_federation(
        rows,
        label_getter=_row_label,
        bootstrap_ratio=bootstrap_ratio,
        client_count=client_count,
        seed=seed,
        shard_policy=shard_policy,
    )
    return FederatedDatasetSplit(
        bootstrap_rows=list(split.bootstrap_items),
        client_shards=_to_client_shards(
            split.client_shards,
            client_pool_split_config=client_pool_split_config,
            seed=seed,
        ),
    )


def split_rows_into_client_shards(
    rows: list[LabeledQueryRow],
    *,
    client_count: int,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
) -> tuple[FederatedClientShard, ...]:
    """heldout validation row를 bootstrap 없이 client shard로 나눈다."""
    assignments = split_items_into_client_shards(
        rows,
        label_getter=_row_label,
        client_count=client_count,
        seed=seed,
        shard_policy=shard_policy,
    )
    return _to_client_shards(assignments)


def _row_label(row: LabeledQueryRow) -> str:
    return str(row["mapped_label_4"])


def _to_client_shards(
    assignments: tuple[FederatedClientShardAssignment[LabeledQueryRow], ...],
    *,
    client_pool_split_config: FederatedClientPoolSplitConfig | None = None,
    seed: int = 0,
) -> tuple[FederatedClientShard, ...]:
    shards: list[FederatedClientShard] = []
    for index, assignment in enumerate(assignments):
        rows = list(assignment.items)
        labeled_rows, unlabeled_rows = _split_client_pool_rows(
            rows,
            client_pool_split_config=client_pool_split_config,
            seed=seed + index,
        )
        shards.append(
            FederatedClientShard(
                client_id=assignment.client_id,
                rows=rows,
                labeled_rows=labeled_rows,
                unlabeled_rows=unlabeled_rows,
                client_pool_split_enforced=client_pool_split_config is not None,
            )
        )
    return tuple(shards)


def _split_client_pool_rows(
    rows: list[LabeledQueryRow],
    *,
    client_pool_split_config: FederatedClientPoolSplitConfig | None,
    seed: int,
) -> tuple[list[LabeledQueryRow], list[LabeledQueryRow]]:
    if client_pool_split_config is None:
        return [], list(rows)

    rng = random.Random(seed)
    labeled_rows: list[LabeledQueryRow] = []
    unlabeled_rows: list[LabeledQueryRow] = []
    rows_by_label: dict[str, list[LabeledQueryRow]] = defaultdict(list)
    for row in rows:
        rows_by_label[_row_label(row)].append(row)

    for label in sorted(rows_by_label):
        bucket = list(rows_by_label[label])
        rng.shuffle(bucket)
        labeled_count = _resolve_labeled_count(
            bucket_size=len(bucket),
            labeled_ratio=client_pool_split_config.labeled_ratio,
        )
        labeled_rows.extend(bucket[:labeled_count])
        unlabeled_rows.extend(bucket[labeled_count:])
    return labeled_rows, unlabeled_rows


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
