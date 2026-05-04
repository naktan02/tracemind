"""Reusable FL SSL method descriptor tests."""

from __future__ import annotations

import pytest

from methods.federated_ssl.fedavg_pseudo_label.fedavg_pseudo_label import (
    FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)
from methods.federated_ssl.registry import (
    list_federated_ssl_method_descriptors,
    resolve_federated_ssl_method_descriptor,
)


def test_federated_ssl_descriptor_registry_resolves_active_baseline() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedavg_pseudo_label")

    assert descriptor is FEDAVG_PSEUDO_LABEL_DESCRIPTOR
    assert descriptor.implementation_status == "active_runtime"
    assert descriptor.client_trainer_name == "local_training_service"
    assert descriptor.pseudo_labeler_name == "agent_pseudo_label_selection"
    assert descriptor.view_generator_name == "training_example_backend"
    assert descriptor.server_aggregator_name == "round_runtime_aggregation_backend"
    assert descriptor.requires_custom_client_runtime is False
    assert descriptor.requires_custom_server_runtime is False


def test_federated_ssl_descriptor_registry_rejects_unwired_method() -> None:
    with pytest.raises(NotImplementedError, match="descriptor is not wired yet"):
        resolve_federated_ssl_method_descriptor("paper_method_candidate")


def test_federated_ssl_descriptor_registry_lists_unique_descriptors() -> None:
    descriptors = list_federated_ssl_method_descriptors(
        method_names=("fedavg_pseudo_label", "fedavg_pseudo_label")
    )

    assert descriptors == (FEDAVG_PSEUDO_LABEL_DESCRIPTOR,)
