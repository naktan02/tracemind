"""FedMatch parameter partition routing metadata."""

from __future__ import annotations

from dataclasses import dataclass

from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER,
)
from methods.federated_ssl.update_partition import FederatedSslUpdatePartitionPolicy

FEDMATCH_SIGMA_PARTITION = "sigma"
FEDMATCH_PSI_PARTITION = "psi"
FEDMATCH_PUBLISHED_STATE_EXPRESSION = "sigma_plus_psi"


@dataclass(frozen=True, slots=True)
class FedMatchTraceParameterMapping:
    """원본 sigma/psi를 PEFT encoder trainable scope에 매핑한다."""

    original_trainable_scope: str
    trace_trainable_scope: str
    frozen_scope: str
    supervised_partition: str
    unsupervised_partition: str
    published_state_expression: str
    labels_at_client_upload_partitions: tuple[str, ...]
    labels_at_server_upload_partitions: tuple[str, ...]


trace_parameter_mapping = FedMatchTraceParameterMapping(
    original_trainable_scope="ResNet9 Conv/Dense full weights",
    trace_trainable_scope="LoRA adapter tensors plus classifier head tensors",
    frozen_scope="Transformer backbone base weights",
    supervised_partition=FEDMATCH_SIGMA_PARTITION,
    unsupervised_partition=FEDMATCH_PSI_PARTITION,
    published_state_expression=FEDMATCH_PUBLISHED_STATE_EXPRESSION,
    labels_at_client_upload_partitions=(
        FEDMATCH_SIGMA_PARTITION,
        FEDMATCH_PSI_PARTITION,
    ),
    labels_at_server_upload_partitions=(FEDMATCH_PSI_PARTITION,),
)

parameter_routing_policy = FederatedSslUpdatePartitionPolicy(
    policy_name="sigma_psi",
    partition_names=(FEDMATCH_SIGMA_PARTITION, FEDMATCH_PSI_PARTITION),
    parameters={
        "supervised_loss_partition": FEDMATCH_SIGMA_PARTITION,
        "unsupervised_loss_partition": FEDMATCH_PSI_PARTITION,
        "published_state": FEDMATCH_PUBLISHED_STATE_EXPRESSION,
        "trace_trainable_scope": trace_parameter_mapping.trace_trainable_scope,
        "frozen_scope": trace_parameter_mapping.frozen_scope,
    },
)


def upload_partitions_for_scenario(*, scenario_name: str) -> tuple[str, ...]:
    normalized = scenario_name.replace("_", "-")
    if normalized == FEDMATCH_SCENARIO_LABELS_AT_CLIENT:
        return trace_parameter_mapping.labels_at_client_upload_partitions
    if normalized == FEDMATCH_SCENARIO_LABELS_AT_SERVER:
        return trace_parameter_mapping.labels_at_server_upload_partitions
    raise ValueError(f"Unsupported FedMatch scenario: {scenario_name!r}")
