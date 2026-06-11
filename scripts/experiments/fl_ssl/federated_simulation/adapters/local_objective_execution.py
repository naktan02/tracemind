"""FL simulation local objective execution adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from methods.federated_ssl.capabilities.plan import FederatedSslCapabilityPlan
from methods.federated_ssl.hooks.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientRoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
    SimulationRunRequest,
)
from scripts.support.configured_callable import load_configured_callable

LocalObjectiveExecutor = Callable[..., ClientRoundExecution | None]


def run_method_or_manual_local_objective_if_supported(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    capability_plan: FederatedSslCapabilityPlan,
    peer_context: FederatedSslPeerContext | None = None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None = None,
    previous_client_partition_parameters: Mapping[str, Any] | None = None,
    previous_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> ClientRoundExecution | None:
    """현재 지원되는 local objective fast path를 실행한다."""

    for executor_path in request.round_runtime_config.local_objective_executors:
        execution = _load_local_objective_executor(executor_path)(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
            round_id=round_id,
            shard=shard,
            training_task=training_task,
            capability_plan=capability_plan,
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            previous_client_partition_parameters=previous_client_partition_parameters,
            previous_query_ssl_algorithm_state=previous_query_ssl_algorithm_state,
        )
        if execution is not None:
            return execution
    return None


def _load_local_objective_executor(executor_path: str) -> LocalObjectiveExecutor:
    return load_configured_callable(
        executor_path,
        field_name="round_runtime.local_objective_executors entries",
    )
