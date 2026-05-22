"""Federated simulation row shard adapter."""

from __future__ import annotations

from methods.federated.client_split import split_client_pool_items
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
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def split_rows_for_federation(
    rows: list[LabeledQueryRow],
    *,
    bootstrap_ratio: float,
    client_count: int,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
    client_pool_split_config: FederatedClientPoolSplitConfig | None = None,
) -> FederatedDatasetSplit:
    """train row를 bootstrap holdout과 non-IID client shard로 나눈다."""
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
        if client_pool_split_config is None:
            labeled_rows, unlabeled_rows = [], list(rows)
        else:
            labeled_rows, unlabeled_rows = split_client_pool_items(
                rows,
                labeled_ratio=client_pool_split_config.labeled_ratio,
                seed=seed + index,
                label_getter=_row_label,
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
