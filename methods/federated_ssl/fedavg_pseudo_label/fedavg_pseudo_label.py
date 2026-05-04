"""FedAvg pseudo-label FL SSL method descriptor."""

from __future__ import annotations

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.registry import register_federated_ssl_method_descriptor

FEDAVG_PSEUDO_LABEL_DESCRIPTOR = FederatedSslMethodDescriptor(
    name="fedavg_pseudo_label",
    implementation_status="active_runtime",
    client_trainer_name="local_training_service",
    pseudo_labeler_name="ssl_pseudo_label_selection_hook",
    view_generator_name="training_example_backend",
    server_aggregator_name="round_runtime_aggregation_backend",
    round_policy_name="round_active_pair_only",
    requires_custom_client_runtime=False,
    requires_custom_server_runtime=False,
)

register_federated_ssl_method_descriptor(
    "fedavg_pseudo_label",
    descriptor=FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)
