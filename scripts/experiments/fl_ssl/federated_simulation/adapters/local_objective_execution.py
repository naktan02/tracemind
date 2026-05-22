"""FL simulation local objective execution adapter."""

from __future__ import annotations

from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientRoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.query_ssl_client_round import (
    run_query_ssl_client_round_if_supported,
)


def run_method_or_manual_local_objective_if_supported(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
) -> ClientRoundExecution | None:
    """현재 지원되는 local objective fast path를 실행한다."""

    return run_query_ssl_client_round_if_supported(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
    )
