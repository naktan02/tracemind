"""Reusable federated shard policy method tests."""

from __future__ import annotations

import random

import pytest

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated.shard_policy.split import (
    sample_dirichlet_counts,
    split_items_for_federation,
)


def _label(item: tuple[str, str]) -> str:
    return item[1]


def test_label_dominant_policy_assigns_items_without_script_row_shape() -> None:
    items = [
        ("a1", "anxiety"),
        ("a2", "anxiety"),
        ("a3", "anxiety"),
        ("a4", "anxiety"),
    ]

    split = split_items_for_federation(
        items,
        label_getter=_label,
        bootstrap_ratio=0.25,
        client_count=2,
        seed=42,
        shard_policy=FederatedShardPolicyConfig(
            name="label_dominant",
            dominant_ratio=0.5,
            client_id_prefix="agent",
        ),
    )

    assert len(split.bootstrap_items) == 1
    assert [shard.client_id for shard in split.client_shards] == [
        "agent_01",
        "agent_02",
    ]
    assert [len(shard.items) for shard in split.client_shards] == [2, 1]


def test_dirichlet_policy_is_deterministic_for_seed() -> None:
    items = [(f"a{index}", "anxiety") for index in range(20)] + [
        (f"n{index}", "normal") for index in range(20)
    ]
    policy = FederatedShardPolicyConfig(
        name="dirichlet_label_skew",
        alpha=0.3,
        client_id_prefix="agent",
    )

    first = split_items_for_federation(
        items,
        label_getter=_label,
        bootstrap_ratio=0.2,
        client_count=5,
        seed=42,
        shard_policy=policy,
    )
    second = split_items_for_federation(
        items,
        label_getter=_label,
        bootstrap_ratio=0.2,
        client_count=5,
        seed=42,
        shard_policy=policy,
    )

    assert first == second
    input_ids = {item[0] for item in items}
    output_ids = {item[0] for item in first.bootstrap_items} | {
        item[0] for shard in first.client_shards for item in shard.items
    }
    assert output_ids == input_ids


def test_sample_dirichlet_counts_preserves_total() -> None:
    counts = sample_dirichlet_counts(
        total=17,
        parts=4,
        alpha=0.3,
        rng=random.Random(7),
    )

    assert sum(counts) == 17
    assert len(counts) == 4


def test_shard_policy_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="dominant_ratio"):
        split_items_for_federation(
            [("a1", "anxiety")],
            label_getter=_label,
            bootstrap_ratio=0.5,
            client_count=2,
            seed=42,
            shard_policy=FederatedShardPolicyConfig(
                name="label_dominant",
                dominant_ratio=None,
                client_id_prefix="agent",
            ),
        )
