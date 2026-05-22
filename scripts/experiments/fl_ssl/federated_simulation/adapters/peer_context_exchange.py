"""FL simulation peer context exchange adapter."""

from __future__ import annotations

from methods.federated_ssl.capability_plan import (
    PEER_CONTEXT_NONE,
    FederatedSslCapabilityPlan,
)


def require_supported_peer_context(
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """v1 simulation에서 지원되는 peer context인지 확인한다."""

    if capability_plan.peer_context_policy_name == PEER_CONTEXT_NONE:
        return
    raise NotImplementedError(
        "peer_context_policy is declared but not implemented in simulation runtime: "
        f"{capability_plan.peer_context_policy_name}"
    )
