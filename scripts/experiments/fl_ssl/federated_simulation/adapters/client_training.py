"""FL simulation client local training adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    FederatedSslSimulationRuntime,
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

from .local_objective_execution import run_method_or_manual_local_objective_if_supported


def build_round_training_scoring_service(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    training_task: Any,
) -> Any:
    """prototype-scored fallback 제거 후에는 별도 scoring service를 만들지 않는다."""

    del request, active, training_task
    return None


def run_client_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    training_scoring_service: Any,
    peer_context: FederatedSslPeerContext | None = None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None = None,
) -> ClientRoundExecution:
    """client shard 하나의 local training을 실행하고 update를 제출한다."""

    query_ssl_execution = run_method_or_manual_local_objective_if_supported(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
        peer_context=peer_context,
        peer_snapshots=peer_snapshots,
    )
    if query_ssl_execution is not None:
        return query_ssl_execution

    descriptor = ssl_method_runtime.descriptor
    if (
        descriptor is not None
        and descriptor.runtime_capabilities.requires_custom_client_runtime
    ):
        raise NotImplementedError(
            "Federated SSL method requires a custom client runtime that is not "
            f"wired for this adapter family: {descriptor.name}"
        )

    raise NotImplementedError(
        "FL SSL simulation no longer supports prototype-scored generic local "
        "training. Use the LoRA-classifier method/manual local objective path."
    )
