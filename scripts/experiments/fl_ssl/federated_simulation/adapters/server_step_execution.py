"""FL simulation server step execution adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from methods.federated_ssl.capability_plan import (
    SERVER_STEP_NONE,
    FederatedSslCapabilityPlan,
)
from scripts.configured_callable import load_configured_callable
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)


def require_supported_server_step(
    capability_plan: FederatedSslCapabilityPlan,
    *,
    server_step_executor: str | None,
) -> None:
    """v1 simulation에서 지원되는 server step인지 확인한다."""

    if capability_plan.server_step_policy_name == SERVER_STEP_NONE:
        return
    if server_step_executor:
        _load_server_step_executor(server_step_executor)
        return
    raise NotImplementedError(
        "selected update family has no simulation executor for server_step_policy: "
        f"{capability_plan.server_step_policy_name}"
    )


@dataclass(frozen=True, slots=True)
class ServerStepExecution:
    """round open 전에 실행된 server-side step 결과."""

    active: ActiveSimulationState
    metrics: dict[str, float]
    model_revision: str | None = None


def run_server_step_if_supported(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    capability_plan: FederatedSslCapabilityPlan,
    round_index: int,
) -> ServerStepExecution:
    """capability plan에 선언된 server-side step을 round open 전에 실행한다."""

    if capability_plan.server_step_policy_name == SERVER_STEP_NONE:
        return ServerStepExecution(active=active, metrics={})
    if not request.server_step_executor:
        raise NotImplementedError(
            "selected update family has no simulation executor for server_step_policy: "
            f"runtime: {capability_plan.server_step_policy_name}"
        )
    executor = _load_server_step_executor(request.server_step_executor)
    return executor(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_index=round_index,
    )


def _load_server_step_executor(executor_path: str) -> Any:
    return load_configured_callable(
        executor_path,
        field_name="round_runtime.server_step_executors entry",
    )
