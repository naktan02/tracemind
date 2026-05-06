"""FedAvg pseudo-label FL SSL method descriptor."""

from __future__ import annotations

from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslRequiredViews,
    FederatedSslRuntimeCapabilities,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.registry import register_federated_ssl_method_descriptor

FEDAVG_PSEUDO_LABEL_DESCRIPTOR = FederatedSslMethodDescriptor(
    name="fedavg_pseudo_label",
    implementation_status="active_runtime",
    required_views=FederatedSslRequiredViews(
        view_names=("single_view",),
        view_generator_name="training_example_backend",
    ),
    local_step=FederatedSslLocalStepSpec(
        step_name="pseudo_label_self_training",
        client_trainer_name="local_training_service",
        pseudo_labeler_name="ssl_pseudo_label_selection_hook",
    ),
    server_step=FederatedSslServerStepSpec(
        server_aggregator_name="round_runtime_aggregation_backend",
        round_policy_name="round_active_pair_only",
        server_aggregate_hint="use_round_runtime_aggregation_backend",
    ),
    runtime_capabilities=FederatedSslRuntimeCapabilities(
        simulation_supported=True,
        live_agent_supported=True,
        live_server_supported=True,
    ),
)

register_federated_ssl_method_descriptor(
    "fedavg_pseudo_label",
    descriptor=FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)
