"""Test-only FL SSL method extension fixture.

이 fixture는 production `methods/federated_ssl/`에 장난감 method를 남기지 않고,
method-local descriptor/recipe/registration 경계를 검증하기 위해 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslMethodRecipe,
    FederatedSslProfileCombination,
    FederatedSslRequiredViews,
    FederatedSslRoundStateExchangeSpec,
    FederatedSslRuntimeCapabilities,
    FederatedSslRuntimePair,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.registry import register_federated_ssl_method_descriptor
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
)


@dataclass(frozen=True, slots=True)
class TestOnlyLocalObjective:
    """test-only method-local local objective seam."""

    objective_name: str = "dummy_metric_weighted_pseudo_label"
    trainer_hint: str = "local_training_service"
    pseudo_labeler_hint: str = "ssl_pseudo_label_selection_hook"


@dataclass(frozen=True, slots=True)
class TestOnlyServerPolicy:
    """test-only method-local server policy seam."""

    policy_name: str = "round_runtime_aggregation_backend"
    aggregation_hint: str = "use_round_runtime_aggregation_backend"
    custom_server_runtime_required: bool = False


@dataclass(frozen=True, slots=True)
class TestOnlyRoundPolicy:
    """test-only method-local round policy seam."""

    policy_name: str = "dummy_metric_weighted_round_policy"
    custom_round_policy_required: bool = True


DUMMY_LOCAL_OBJECTIVE = TestOnlyLocalObjective()
DUMMY_SERVER_POLICY = TestOnlyServerPolicy()
DUMMY_ROUND_POLICY = TestOnlyRoundPolicy()

DUMMY_FEDERATED_SSL_RECIPE = FederatedSslMethodRecipe(
    method_name="dummy_metric_weighted_ssl",
    supported_local_update_profile_names=("dummy_local_update_profile_v1",),
    supported_runtime_pairs=(
        FederatedSslRuntimePair(
            adapter_family_name=DIAGONAL_SCALE_ADAPTER_KIND,
            aggregation_backend_name="fedavg",
        ),
    ),
    supported_profile_combinations=(
        FederatedSslProfileCombination(
            local_update_profile_name="dummy_local_update_profile_v1",
            round_runtime_profile_name="dummy_round_runtime_profile_v1",
        ),
    ),
)

DUMMY_FEDERATED_SSL_DESCRIPTOR = FederatedSslMethodDescriptor(
    name="dummy_metric_weighted_ssl",
    implementation_status="test_only",
    required_views=FederatedSslRequiredViews(
        view_names=("single_view",),
        view_generator_name="training_example_backend",
    ),
    local_step=FederatedSslLocalStepSpec(
        step_name=DUMMY_LOCAL_OBJECTIVE.objective_name,
        client_trainer_name=DUMMY_LOCAL_OBJECTIVE.trainer_hint,
        pseudo_labeler_name=DUMMY_LOCAL_OBJECTIVE.pseudo_labeler_hint,
    ),
    server_step=FederatedSslServerStepSpec(
        server_aggregator_name=DUMMY_SERVER_POLICY.policy_name,
        round_policy_name=DUMMY_ROUND_POLICY.policy_name,
        server_aggregate_hint=DUMMY_SERVER_POLICY.aggregation_hint,
    ),
    round_state_exchange=FederatedSslRoundStateExchangeSpec(
        exchange_name="client_metric_summary",
        required_client_metric_keys=("mean_confidence",),
        summary_metric_prefix="dummy_round_state",
        requires_custom_exchange=True,
    ),
    runtime_capabilities=FederatedSslRuntimeCapabilities(
        simulation_supported=True,
        live_agent_supported=False,
        live_server_supported=False,
        requires_custom_server_runtime=(
            DUMMY_SERVER_POLICY.custom_server_runtime_required
            or DUMMY_ROUND_POLICY.custom_round_policy_required
        ),
    ),
    recipe=DUMMY_FEDERATED_SSL_RECIPE,
)

descriptor = register_federated_ssl_method_descriptor(
    "dummy_metric_weighted_ssl",
)(DUMMY_FEDERATED_SSL_DESCRIPTOR)
local_objective = DUMMY_LOCAL_OBJECTIVE
server_policy = DUMMY_SERVER_POLICY
round_policy = DUMMY_ROUND_POLICY
recipe = DUMMY_FEDERATED_SSL_RECIPE
