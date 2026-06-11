"""Live FSSL task context parsing tests."""

from __future__ import annotations

import pytest

from methods.federated_ssl.live_task_context import (
    build_method_config_from_live_fssl_context,
    build_peer_context_from_live_fssl_context,
)


def test_live_fssl_context_builds_peer_context_for_matching_client() -> None:
    context = {
        "schema_version": "fssl_context.v1",
        "method_name": "fedmatch",
        "context_kind": "peer_context",
        "peer_context": {
            "schema_version": "peer_context_task.v1",
            "policy_name": "fixed_probe_output_knn",
            "source_round_id": "round_prev",
            "round_index_zero_based": 2,
            "warmup": False,
            "summary_metrics": {"fedmatch.update_count": 2.0},
            "client_contexts": [
                {
                    "client_id": "agent_01",
                    "helper_client_ids": ["agent_02", "agent_03"],
                }
            ],
        },
    }

    peer_context = build_peer_context_from_live_fssl_context(
        fssl_context=context,
        client_id="agent_01",
    )

    assert peer_context is not None
    assert peer_context.policy_name == "fixed_probe_output_knn"
    assert peer_context.round_index_zero_based == 2
    assert peer_context.helper_client_ids == ("agent_02", "agent_03")
    assert peer_context.refreshed is True
    assert peer_context.metadata["source_round_id"] == "round_prev"
    assert peer_context.metadata["summary_metrics"] == {
        "fedmatch.update_count": 2.0
    }


def test_live_fssl_context_uses_default_peer_policy() -> None:
    peer_context = build_peer_context_from_live_fssl_context(
        fssl_context={"peer_context": {"warmup": True}},
        client_id="agent_01",
        default_policy_name="fixed_probe_output_knn",
    )

    assert peer_context is not None
    assert peer_context.policy_name == "fixed_probe_output_knn"
    assert peer_context.helper_client_ids == ()
    assert peer_context.refreshed is False


def test_live_fssl_context_builds_method_config_with_peer_scenario() -> None:
    method_config = build_method_config_from_live_fssl_context(
        fssl_method="fedmatch",
        fssl_context={
            "method_config": {"parameter_overrides": {"lambda_s": 0.5}},
            "peer_context": {"scenario": "labels-at-server"},
        },
    )

    assert method_config["name"] == "fedmatch"
    assert method_config["scenario"] == "labels-at-server"
    assert method_config["parameter_overrides"] == {"lambda_s": 0.5}


def test_live_fssl_context_requires_method_name_for_method_config() -> None:
    with pytest.raises(ValueError, match="fssl_method"):
        build_method_config_from_live_fssl_context(
            fssl_method=None,
            fssl_context={},
        )
