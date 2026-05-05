"""Federated simulation row shard adapter."""

from __future__ import annotations

from methods.federated.shard_policy.base import (
    FederatedClientShardAssignment,
    FederatedShardPolicyConfig,
)
from methods.federated.shard_policy.split import (
    split_items_for_federation,
    split_items_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
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
        client_shards=_to_client_shards(split.client_shards),
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
) -> tuple[FederatedClientShard, ...]:
    return tuple(
        FederatedClientShard(
            client_id=assignment.client_id,
            rows=list(assignment.items),
        )
        for assignment in assignments
    )
