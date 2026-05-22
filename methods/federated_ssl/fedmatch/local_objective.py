"""FedMatch local objective metadata."""

from __future__ import annotations

from methods.federated_ssl.local_objective import FederatedSslLocalObjectiveSpec

FEDMATCH_LOCAL_OBJECTIVE_NAME = "fedmatch_sigma_psi_local_objective"

local_objective_spec = FederatedSslLocalObjectiveSpec(
    objective_name=FEDMATCH_LOCAL_OBJECTIVE_NAME,
    required_batch_views=("weak_text", "strong_text"),
    metric_prefix="fedmatch_local",
    parameters={
        "supervised_partition": "sigma",
        "unsupervised_partition": "psi",
        "agreement_loss": "helper_consistency",
    },
)
