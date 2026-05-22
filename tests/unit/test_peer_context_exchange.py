"""FL SSL peer context exchange unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    peer_context_exchange,
)


def test_prediction_similarity_peer_context_is_supported_capability() -> None:
    peer_context_exchange.require_supported_peer_context(_capability_plan("none"))
    peer_context_exchange.require_supported_peer_context(
        _capability_plan("prediction_similarity_topk")
    )


def test_prediction_similarity_peer_context_selects_nearest_on_refresh_round() -> None:
    contexts = peer_context_exchange.build_peer_context_by_client(
        capability_plan=_capability_plan("prediction_similarity_topk"),
        ssl_method_config=SimpleNamespace(
            round_state_exchange={"num_helpers": 2, "refresh_interval": 10}
        ),
        selected_client_ids=("client_a", "client_b"),
        round_index=10,
        client_vectors={
            "client_a": (0.0, 0.0),
            "client_b": (0.2, 0.0),
            "client_c": (3.0, 0.0),
            "client_d": (0.1, 0.0),
        },
    )

    assert contexts["client_a"].helper_client_ids == ("client_d", "client_b")
    assert contexts["client_a"].refreshed is True
    assert contexts["client_a"].metadata["refresh_due"] is True
    assert contexts["client_a"].metadata["parameter_source"] == "effective_parameters"
    assert contexts["client_a"].metadata["selection_index_backend"] in {
        "scipy_kdtree",
        "full_scan",
    }
    assert contexts["client_a"].metadata["selection_query_size"] == 3


def test_prediction_similarity_peer_context_respects_refresh_interval() -> None:
    contexts = peer_context_exchange.build_peer_context_by_client(
        capability_plan=_capability_plan("prediction_similarity_topk"),
        ssl_method_config=SimpleNamespace(
            round_state_exchange={"num_helpers": 2, "refresh_interval": 10}
        ),
        selected_client_ids=("client_a",),
        round_index=9,
        client_vectors={
            "client_a": (0.0, 0.0),
            "client_b": (0.2, 0.0),
        },
    )

    assert contexts["client_a"].helper_client_ids == ()
    assert contexts["client_a"].refreshed is False
    assert contexts["client_a"].metadata["refresh_due"] is False


def test_prediction_similarity_peer_context_uses_method_parameter_overrides() -> None:
    contexts = peer_context_exchange.build_peer_context_by_client(
        capability_plan=_capability_plan("prediction_similarity_topk"),
        ssl_method_config=SimpleNamespace(
            round_state_exchange={"num_helpers": 2, "refresh_interval": 10},
            effective_parameters={
                "num_helpers": 1,
                "helper_refresh_interval": 1,
            },
        ),
        selected_client_ids=("client_a",),
        round_index=2,
        client_vectors={
            "client_a": (0.0, 0.0),
            "client_b": (0.2, 0.0),
            "client_c": (0.1, 0.0),
        },
    )

    assert contexts["client_a"].helper_client_ids == ("client_c",)
    assert contexts["client_a"].metadata["num_helpers"] == 1
    assert contexts["client_a"].metadata["refresh_interval"] == 1


def test_prediction_similarity_peer_context_requires_method_parameters() -> None:
    with pytest.raises(ValueError, match="ssl_method_config"):
        peer_context_exchange.build_peer_context_by_client(
            capability_plan=_capability_plan("prediction_similarity_topk"),
            ssl_method_config=None,
            selected_client_ids=("client_a",),
            round_index=1,
        )


def _capability_plan(peer_context_policy_name: str) -> FederatedSslCapabilityPlan:
    return FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy=None,
        aggregation_weight_policy=None,
        labeled_exposure_policy=None,
        local_supervision_regime=None,
        server_step_policy=None,
        peer_context_policy={"name": peer_context_policy_name},
        update_partition_policy=None,
        query_multiview_source=None,
    )
