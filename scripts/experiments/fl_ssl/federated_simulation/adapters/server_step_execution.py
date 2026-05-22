"""FL simulation server step execution adapter."""

from __future__ import annotations

from methods.federated_ssl.capability_plan import (
    SERVER_STEP_NONE,
    FederatedSslCapabilityPlan,
)


def require_supported_server_step(
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """v1 simulation에서 지원되는 server step인지 확인한다."""

    if capability_plan.server_step_policy_name == SERVER_STEP_NONE:
        return
    raise NotImplementedError(
        "server_step_policy is declared but not implemented in simulation runtime: "
        f"{capability_plan.server_step_policy_name}"
    )
