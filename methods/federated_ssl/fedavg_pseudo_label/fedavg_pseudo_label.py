"""FedAvg pseudo-label FL SSL method registry wiring."""

from __future__ import annotations

from methods.federated_ssl.fedavg_pseudo_label.descriptor import (
    FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)
from methods.federated_ssl.registry import register_federated_ssl_method_descriptor

register_federated_ssl_method_descriptor(
    "fedavg_pseudo_label",
)(FEDAVG_PSEUDO_LABEL_DESCRIPTOR)
