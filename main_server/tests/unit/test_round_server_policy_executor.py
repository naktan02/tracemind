"""server policy runtime capability 테스트."""

from __future__ import annotations

import pytest

from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundValidationError,
)
from main_server.src.services.federation.rounds.server_policy.executor import (
    DefaultServerPolicyExecutor,
)
from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslRequiredViews,
    FederatedSslRuntimeCapabilities,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.fedavg_pseudo_label.descriptor import (
    FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)


def _build_test_descriptor(
    *,
    server_aggregator_name: str = "round_runtime_aggregation_backend",
    round_policy_name: str = "round_active_pair_only",
    server_aggregate_hint: str = "use_round_runtime_aggregation_backend",
    live_server_supported: bool = True,
    requires_custom_server_runtime: bool = False,
) -> FederatedSslMethodDescriptor:
    return FederatedSslMethodDescriptor(
        name="test_server_policy_method",
        implementation_status="test_only",
        required_views=FederatedSslRequiredViews(
            view_names=("single_view",),
            view_generator_name="training_example_backend",
        ),
        local_step=FederatedSslLocalStepSpec(
            step_name="test_local_step",
            client_trainer_name="test_client_trainer",
            pseudo_labeler_name="test_pseudo_labeler",
        ),
        server_step=FederatedSslServerStepSpec(
            server_aggregator_name=server_aggregator_name,
            round_policy_name=round_policy_name,
            server_aggregate_hint=server_aggregate_hint,
        ),
        runtime_capabilities=FederatedSslRuntimeCapabilities(
            simulation_supported=True,
            live_agent_supported=True,
            live_server_supported=live_server_supported,
            requires_custom_server_runtime=requires_custom_server_runtime,
        ),
    )


def test_default_server_policy_executor_accepts_round_runtime_policy() -> None:
    summary = DefaultServerPolicyExecutor().prepare_finalize(
        method_descriptor=FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
        round_id="round_001",
        update_count=2,
    )

    assert summary.method_name == "fedavg_pseudo_label"
    assert summary.round_id == "round_001"
    assert summary.server_aggregator_name == "round_runtime_aggregation_backend"
    assert summary.round_policy_name == "round_active_pair_only"
    assert summary.server_aggregate_hint == "use_round_runtime_aggregation_backend"
    assert summary.update_count == 2


def test_default_server_policy_executor_rejects_custom_server_runtime() -> None:
    descriptor = _build_test_descriptor(requires_custom_server_runtime=True)

    with pytest.raises(RoundValidationError, match="custom server runtime"):
        DefaultServerPolicyExecutor().prepare_finalize(
            method_descriptor=descriptor,
            round_id="round_001",
            update_count=1,
        )


def test_default_server_policy_executor_rejects_custom_server_policy_name() -> None:
    descriptor = _build_test_descriptor(server_aggregator_name="custom_policy")

    with pytest.raises(RoundValidationError, match="Unsupported server aggregation"):
        DefaultServerPolicyExecutor().prepare_finalize(
            method_descriptor=descriptor,
            round_id="round_001",
            update_count=1,
        )
