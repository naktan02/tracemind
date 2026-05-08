"""FedAvg pseudo-label FL SSL method registry wiring."""

from __future__ import annotations

from methods.federated_ssl.fedavg_pseudo_label.descriptor import (
    FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
    FEDAVG_PSEUDO_LABEL_RECIPE,
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
from methods.federated_ssl.registry import register_federated_ssl_method_descriptor

descriptor = FEDAVG_PSEUDO_LABEL_DESCRIPTOR
local_objective = FEDAVG_PSEUDO_LABEL_LOCAL_OBJECTIVE
server_policy = FEDAVG_PSEUDO_LABEL_SERVER_POLICY
round_policy = FEDAVG_PSEUDO_LABEL_ROUND_POLICY
recipe = FEDAVG_PSEUDO_LABEL_RECIPE

register_federated_ssl_method_descriptor(
    "fedavg_pseudo_label",
)(FEDAVG_PSEUDO_LABEL_DESCRIPTOR)
