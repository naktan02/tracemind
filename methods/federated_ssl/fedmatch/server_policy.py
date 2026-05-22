"""FedMatch server policy metadata."""

from __future__ import annotations

from methods.federated_ssl.capability_axes import (
    SERVER_UPDATE_FEDAVG_MERGED_DELTA,
    SERVER_UPDATE_FEDMATCH_PARTITIONED,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER,
)
from methods.federated_ssl.fedmatch.parameter_routing import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
)
from methods.federated_ssl.server_step import FederatedSslServerStepPolicy

FEDMATCH_LABELS_AT_CLIENT_POLICY = "fedmatch_labels_at_client"
FEDMATCH_LABELS_AT_SERVER_POLICY = "fedmatch_labels_at_server"

labels_at_client_policy = FederatedSslServerStepPolicy(
    policy_name=FEDMATCH_LABELS_AT_CLIENT_POLICY,
    parameters={
        "scenario": FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
        "server_step_policy": "none",
        "current_server_update_policy": SERVER_UPDATE_FEDAVG_MERGED_DELTA,
        "target_server_update_policy": SERVER_UPDATE_FEDMATCH_PARTITIONED,
        "aggregated_partitions": (FEDMATCH_SIGMA_PARTITION, FEDMATCH_PSI_PARTITION),
        "aggregation_weight_policy": "uniform",
    },
)
labels_at_server_policy = FederatedSslServerStepPolicy(
    policy_name=FEDMATCH_LABELS_AT_SERVER_POLICY,
    parameters={
        "scenario": FEDMATCH_SCENARIO_LABELS_AT_SERVER,
        "server_step_policy": "supervised_seed_step",
        "server_update_policy": SERVER_UPDATE_FEDMATCH_PARTITIONED,
        "server_trainable_partition": FEDMATCH_SIGMA_PARTITION,
        "client_aggregated_partitions": (FEDMATCH_PSI_PARTITION,),
        "aggregation_weight_policy": "uniform",
    },
)
