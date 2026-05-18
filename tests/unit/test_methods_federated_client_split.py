"""Federated client split policy helper tests."""

from __future__ import annotations

import pytest

from methods.federated.client_split import (
    LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT,
    LABELED_EXPOSURE_SERVER_ONLY_SEED,
    LABELED_EXPOSURE_SHARED_CLIENT_SEED,
    FederatedLabeledExposurePolicy,
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
    assert shared_client.exposes_client_labeled_rows is True
    assert shared_client.shares_same_labeled_rows_across_clients is True
    assert shared_client.storage_group_name == "shared_client_labeled"
    assert server_only.exposes_client_labeled_rows is False
    assert server_only.shares_same_labeled_rows_across_clients is False
    assert server_only.storage_group_name == "server_only_labeled"
    assert server_only.to_payload() == {"name": "server_only_seed"}


def test_labeled_exposure_policy_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="labeled_exposure_policy.name"):
        FederatedLabeledExposurePolicy(name="unknown")
