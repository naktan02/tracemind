"""FedMatch server policy metadata."""

from __future__ import annotations

from methods.federated_ssl.server_step import FederatedSslServerStepPolicy

FEDMATCH_LABELS_AT_CLIENT_POLICY = "fedmatch_labels_at_client"
FEDMATCH_LABELS_AT_SERVER_POLICY = "fedmatch_labels_at_server"

labels_at_client_policy = FederatedSslServerStepPolicy(
    policy_name=FEDMATCH_LABELS_AT_CLIENT_POLICY,
    parameters={"server_step_policy": "none"},
)
labels_at_server_policy = FederatedSslServerStepPolicy(
    policy_name=FEDMATCH_LABELS_AT_SERVER_POLICY,
    parameters={"server_step_policy": "supervised_seed_step"},
)
