"""FedMatch sigma/psi partitioning metadata and runtime plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.federated_ssl.capabilities.plan import (
    LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
)
from methods.federated_ssl.fedmatch.local_objective import (
    FedMatchLocalObjectiveParameters,
    FedMatchPartitionedTensorObjective,
    build_fedmatch_partitioned_tensor_objective,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER,
)
from methods.federated_ssl.hooks.partitioned_update import (
    FederatedSslPartitionedUpdateHook,
)
from methods.federated_ssl.local_supervision import (
    FederatedSslLocalSupervisionRegime,
    resolve_local_supervision_regime,
)

FEDMATCH_SIGMA_PARTITION = "sigma"
FEDMATCH_PSI_PARTITION = "psi"
FEDMATCH_PARTITION_NAMES = (FEDMATCH_SIGMA_PARTITION, FEDMATCH_PSI_PARTITION)
FEDMATCH_PARTITION_POLICY_NAME = "sigma_psi"
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


@dataclass(frozen=True, slots=True)
class FedMatchPartitionedRuntimePlan:
    """FedMatch sigma/psi 의미를 update-family runtime 입력으로 고정한다."""

    scenario_name: str
    local_supervision_regime: FederatedSslLocalSupervisionRegime
    parameters: FedMatchLocalObjectiveParameters
    physical_objective: FedMatchPartitionedTensorObjective
    sequential_objective: FedMatchPartitionedTensorObjective
    partition_hook: FederatedSslPartitionedUpdateHook
    partition_names: tuple[str, ...]
    supervised_partition: str
    unsupervised_partition: str
    upload_partitions: tuple[str, ...]
    l1_sparse_partitions: tuple[str, ...]
    psi_factor: float
    metric_prefix: str = "fedmatch"

    @property
    def emit_supervised_partition(self) -> bool:
        return self.supervised_partition in self.upload_partitions

    @property
    def diagnostic_acceptance_threshold(self) -> float:
        return self.parameters.confidence_threshold


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


def build_fedmatch_partitioned_runtime_plan(
    *,
    scenario_name: str | None,
    effective_parameters: Mapping[str, object],
) -> FedMatchPartitionedRuntimePlan:
    """FedMatch method config를 partitioned adapter runtime plan으로 정규화한다."""

    normalized_scenario = normalize_fedmatch_scenario_name(scenario_name)
    parameters = FedMatchLocalObjectiveParameters.from_mapping(effective_parameters)
    partition_hook = build_fedmatch_partitioned_update_hook(
        scenario_name=normalized_scenario,
    )
    return FedMatchPartitionedRuntimePlan(
        scenario_name=normalized_scenario,
        local_supervision_regime=resolve_fedmatch_local_supervision_regime(
            normalized_scenario
        ),
        parameters=parameters,
        physical_objective=build_fedmatch_partitioned_tensor_objective(parameters),
        sequential_objective=build_fedmatch_partitioned_tensor_objective(
            parameters,
            omit_regularization_for_single_trainable_model=True,
        ),
        partition_hook=partition_hook,
        partition_names=partition_hook.partition_names,
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        upload_partitions=partition_hook.upload_partitions,
        l1_sparse_partitions=partition_hook.l1_sparse_partitions,
        psi_factor=resolve_fedmatch_psi_factor(effective_parameters),
    )


def build_fedmatch_partitioned_update_hook(
    *,
    scenario_name: str,
) -> FederatedSslPartitionedUpdateHook:
    """FedMatch partition 요구사항을 공통 FSSL hook surface로 변환한다."""

    upload_partitions = upload_partitions_for_scenario(scenario_name=scenario_name)
    return FederatedSslPartitionedUpdateHook(
        hook_name="partitioned_update",
        partition_names=FEDMATCH_PARTITION_NAMES,
        upload_partitions=upload_partitions,
        aggregate_partitions=upload_partitions,
        l1_sparse_partitions=(FEDMATCH_PSI_PARTITION,),
        published_state_expression=FEDMATCH_PUBLISHED_STATE_EXPRESSION,
        parameters={
            "policy_name": FEDMATCH_PARTITION_POLICY_NAME,
            "supervised_partition": FEDMATCH_SIGMA_PARTITION,
            "unsupervised_partition": FEDMATCH_PSI_PARTITION,
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


def normalize_fedmatch_scenario_name(scenario_name: str | None) -> str:
    normalized = (scenario_name or FEDMATCH_SCENARIO_LABELS_AT_CLIENT).replace(
        "_",
        "-",
    )
    if normalized not in {
        FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
        FEDMATCH_SCENARIO_LABELS_AT_SERVER,
    }:
        raise ValueError(f"Unsupported FedMatch scenario: {scenario_name!r}.")
    return normalized


def resolve_fedmatch_local_supervision_regime(
    scenario_name: str,
) -> FederatedSslLocalSupervisionRegime:
    if scenario_name == FEDMATCH_SCENARIO_LABELS_AT_CLIENT:
        return resolve_local_supervision_regime(
            LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED
        )
    if scenario_name == FEDMATCH_SCENARIO_LABELS_AT_SERVER:
        return resolve_local_supervision_regime(LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY)
    raise ValueError(f"Unsupported FedMatch scenario: {scenario_name!r}.")


def resolve_fedmatch_psi_factor(parameters: Mapping[str, object]) -> float:
    value = parameters.get("psi_factor", 0.2)
    factor = float(value)
    if factor < 0.0:
        raise ValueError("FedMatch psi_factor must not be negative.")
    return factor
