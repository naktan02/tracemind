"""FedAvg pseudo-label FL SSL method descriptor."""

from __future__ import annotations

from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslMethodRecipe,
    FederatedSslRequiredViews,
    FederatedSslRoundStateExchangeSpec,
    FederatedSslRuntimeCapabilities,
    FederatedSslRuntimePair,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.fedavg_pseudo_label.local_objective import (
    FEDAVG_PSEUDO_LABEL_LOCAL_OBJECTIVE,
)
from methods.federated_ssl.fedavg_pseudo_label.round_policy import (
    FEDAVG_PSEUDO_LABEL_ROUND_POLICY,
)
from methods.federated_ssl.fedavg_pseudo_label.server_policy import (
    FEDAVG_PSEUDO_LABEL_SERVER_POLICY,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
)

FEDAVG_PSEUDO_LABEL_RECIPE = FederatedSslMethodRecipe(
    method_name="fedavg_pseudo_label",
    supported_local_update_profile_names=(
        "prototype_pseudo_label_v1",
        "prototype_top1_confidence_v1",
        "lora_pseudo_label_v1",
    ),
    supported_runtime_pairs=(
        FederatedSslRuntimePair(
            adapter_family_name=DIAGONAL_SCALE_ADAPTER_KIND,
            aggregation_backend_name="fedavg",
        ),
        FederatedSslRuntimePair(
            adapter_family_name=LORA_CLASSIFIER_ADAPTER_KIND,
            aggregation_backend_name="fedavg",
        ),
    ),
)

FEDAVG_PSEUDO_LABEL_DESCRIPTOR = FederatedSslMethodDescriptor(
    name="fedavg_pseudo_label",
    implementation_status="active_runtime",
    required_views=FederatedSslRequiredViews(
        view_names=("single_view",),
        view_generator_name="training_example_backend",
    ),
    local_step=FederatedSslLocalStepSpec(
        step_name=FEDAVG_PSEUDO_LABEL_LOCAL_OBJECTIVE.objective_name,
        client_trainer_name=FEDAVG_PSEUDO_LABEL_LOCAL_OBJECTIVE.trainer_hint,
        pseudo_labeler_name=(FEDAVG_PSEUDO_LABEL_LOCAL_OBJECTIVE.pseudo_labeler_hint),
        training_row_source=(FEDAVG_PSEUDO_LABEL_LOCAL_OBJECTIVE.training_row_source),
    ),
    server_step=FederatedSslServerStepSpec(
        server_aggregator_name=FEDAVG_PSEUDO_LABEL_SERVER_POLICY.policy_name,
        round_policy_name=FEDAVG_PSEUDO_LABEL_ROUND_POLICY.policy_name,
        server_aggregate_hint=FEDAVG_PSEUDO_LABEL_SERVER_POLICY.aggregation_hint,
    ),
    round_state_exchange=FederatedSslRoundStateExchangeSpec(exchange_name="none"),
    runtime_capabilities=FederatedSslRuntimeCapabilities(
        simulation_supported=True,
        live_agent_supported=True,
        live_server_supported=True,
        requires_custom_server_runtime=(
            FEDAVG_PSEUDO_LABEL_SERVER_POLICY.custom_server_runtime_required
            or FEDAVG_PSEUDO_LABEL_ROUND_POLICY.custom_round_policy_required
        ),
    ),
    recipe=FEDAVG_PSEUDO_LABEL_RECIPE,
)
