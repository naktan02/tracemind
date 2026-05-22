"""FedMatch parameter partition routing metadata."""

from __future__ import annotations

from methods.federated_ssl.update_partition import FederatedSslUpdatePartitionPolicy

FEDMATCH_SIGMA_PARTITION = "sigma"
FEDMATCH_PSI_PARTITION = "psi"

parameter_routing_policy = FederatedSslUpdatePartitionPolicy(
    policy_name="sigma_psi",
    partition_names=(FEDMATCH_SIGMA_PARTITION, FEDMATCH_PSI_PARTITION),
    parameters={
        "supervised_loss_partition": FEDMATCH_SIGMA_PARTITION,
        "unsupervised_loss_partition": FEDMATCH_PSI_PARTITION,
        "published_state": "sigma_plus_psi",
    },
)
