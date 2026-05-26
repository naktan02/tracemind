"""FedMatch partitioned runtime plan helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.federated_ssl.capability_plan import (
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
from methods.federated_ssl.fedmatch.parameter_routing import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
    parameter_routing_policy,
    upload_partitions_for_scenario,
)
from methods.federated_ssl.local_supervision import (
    FederatedSslLocalSupervisionRegime,
    resolve_local_supervision_regime,
)


@dataclass(frozen=True, slots=True)
class FedMatchPartitionedRuntimePlan:
    """FedMatch sigma/psi 의미를 adapter-family runtime 입력으로 고정한다."""

    scenario_name: str
    local_supervision_regime: FederatedSslLocalSupervisionRegime
    parameters: FedMatchLocalObjectiveParameters
    physical_objective: FedMatchPartitionedTensorObjective
    sequential_objective: FedMatchPartitionedTensorObjective
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


def build_fedmatch_partitioned_runtime_plan(
    *,
    scenario_name: str | None,
    effective_parameters: Mapping[str, object],
) -> FedMatchPartitionedRuntimePlan:
    """FedMatch method config를 partitioned adapter runtime plan으로 정규화한다."""

    normalized_scenario = normalize_fedmatch_scenario_name(scenario_name)
    parameters = FedMatchLocalObjectiveParameters.from_mapping(effective_parameters)
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
        partition_names=parameter_routing_policy.partition_names,
        supervised_partition=FEDMATCH_SIGMA_PARTITION,
        unsupervised_partition=FEDMATCH_PSI_PARTITION,
        upload_partitions=upload_partitions_for_scenario(
            scenario_name=normalized_scenario
        ),
        l1_sparse_partitions=(FEDMATCH_PSI_PARTITION,),
        psi_factor=resolve_fedmatch_psi_factor(effective_parameters),
    )


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
