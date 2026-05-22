"""Generic partitioned FedAvg strategy identity.

Adapter family modules own the partition materialization and next-state projection.
This package owns only the aggregation backend name required by the registry
convention.
"""

from __future__ import annotations

from methods.federated.aggregation.fedavg.strategy import (
    FedAvgAdapterStrategySpec,
    register_fedavg_adapter_strategy,
)

PARTITIONED_FEDAVG_METHOD_NAME = "partitioned_fedavg"


def register_partitioned_fedavg_adapter_strategy(
    spec: FedAvgAdapterStrategySpec,
) -> None:
    """adapter family projection module에서 partitioned FedAvg strategy를 등록한다."""

    register_fedavg_adapter_strategy(
        FedAvgAdapterStrategySpec(
            adapter_kind=spec.adapter_kind,
            state_type=spec.state_type,
            update_type=spec.update_type,
            context=spec.context,
            aliases=spec.aliases,
            implementation_module=spec.implementation_module,
            core_function_name=spec.core_function_name,
            metadata=spec.metadata,
            aggregate=spec.aggregate,
            method_name=PARTITIONED_FEDAVG_METHOD_NAME,
        )
    )
