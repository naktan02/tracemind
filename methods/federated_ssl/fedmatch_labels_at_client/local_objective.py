"""FedMatch labels-at-client variant는 family local objective core를 재사용한다."""

from __future__ import annotations

from methods.federated_ssl.fedmatch.local_objective import (
    FEDMATCH_LOCAL_OBJECTIVE_NAME,
    FedMatchLocalObjectiveParameters,
    FedMatchPartitionedTensorObjective,
    build_fedmatch_partitioned_tensor_objective,
)

VARIANT_LOCAL_OBJECTIVE_NAME = FEDMATCH_LOCAL_OBJECTIVE_NAME
VariantLocalObjectiveParameters = FedMatchLocalObjectiveParameters
VariantPartitionedTensorObjective = FedMatchPartitionedTensorObjective
build_variant_partitioned_tensor_objective = (
    build_fedmatch_partitioned_tensor_objective
)
