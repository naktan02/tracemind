"""Federated client split policy helper tests."""

from __future__ import annotations

import pytest

from methods.federated.client_split import (
    LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT,
    LABELED_EXPOSURE_SERVER_ONLY_SEED,
    LABELED_EXPOSURE_SHARED_CLIENT_SEED,
    FederatedLabeledExposurePolicy,
    resolve_bootstrap_labeled_rows,
    resolve_client_visible_labeled_rows,
)


def test_labeled_exposure_policy_normalizes_known_modes() -> None:
    client_local = FederatedLabeledExposurePolicy.from_mapping(
        {"name": LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT}
    )
    shared_client = FederatedLabeledExposurePolicy.from_mapping(
        {"name": LABELED_EXPOSURE_SHARED_CLIENT_SEED}
    )
    server_only = FederatedLabeledExposurePolicy.from_mapping(
        {"name": LABELED_EXPOSURE_SERVER_ONLY_SEED}
    )

    assert client_local.exposes_client_labeled_rows is True
    assert client_local.shares_same_labeled_rows_across_clients is False
    assert client_local.storage_group_name == "client_local_labeled"
    assert client_local.compact_slug == "client_local"
    assert shared_client.exposes_client_labeled_rows is True
    assert shared_client.shares_same_labeled_rows_across_clients is True
    assert shared_client.storage_group_name == "shared_client_labeled"
    assert shared_client.compact_slug == "shared_client"
    assert server_only.exposes_client_labeled_rows is False
    assert server_only.shares_same_labeled_rows_across_clients is False
    assert server_only.storage_group_name == "server_only_labeled"
    assert server_only.compact_slug == "server_only"
    assert server_only.to_payload() == {"name": "server_only_seed"}


def test_labeled_exposure_policy_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="labeled_exposure_policy.name"):
        FederatedLabeledExposurePolicy(name="unknown")


def test_labeled_exposure_policy_resolves_runtime_visible_rows() -> None:
    client_local = FederatedLabeledExposurePolicy(
        name=LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT
    )
    shared_client = FederatedLabeledExposurePolicy(
        name=LABELED_EXPOSURE_SHARED_CLIENT_SEED
    )
    server_only = FederatedLabeledExposurePolicy(
        name=LABELED_EXPOSURE_SERVER_ONLY_SEED
    )

    assert resolve_client_visible_labeled_rows(
        policy=client_local,
        client_local_rows=["client-a"],
        shared_seed_rows=["seed-a", "seed-b"],
    ) == ["client-a"]
    assert resolve_client_visible_labeled_rows(
        policy=shared_client,
        client_local_rows=["client-a"],
        shared_seed_rows=["seed-a", "seed-b"],
    ) == ["seed-a", "seed-b"]
    assert (
        resolve_client_visible_labeled_rows(
            policy=server_only,
            client_local_rows=["client-a"],
            shared_seed_rows=["seed-a", "seed-b"],
        )
        == []
    )
    assert resolve_bootstrap_labeled_rows(
        policy=server_only,
        split_bootstrap_rows=["bootstrap-a"],
        shared_seed_rows=["seed-a"],
    ) == ["seed-a"]
    assert resolve_bootstrap_labeled_rows(
        policy=client_local,
        split_bootstrap_rows=["bootstrap-a"],
        shared_seed_rows=["seed-a"],
    ) == ["bootstrap-a"]
