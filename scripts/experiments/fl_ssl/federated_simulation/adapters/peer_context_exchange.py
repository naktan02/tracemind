"""FL simulation peer context exchange adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from methods.federated_ssl.capabilities.plan import (
    PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    PEER_CONTEXT_NONE,
    FederatedSslCapabilityPlan,
)
from methods.federated_ssl.hooks.peer_context import (
    FederatedSslPeerContext,
    build_fixed_probe_peer_context_by_client,
    resolve_fixed_probe_peer_context_parameters,
)


def require_supported_peer_context(
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """v1 simulation에서 지원되는 peer context인지 확인한다."""

    if capability_plan.peer_context_policy_name in {
        PEER_CONTEXT_NONE,
        PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    }:
        return
    raise NotImplementedError(
        "peer_context_policy is declared but not implemented in simulation runtime: "
        f"{capability_plan.peer_context_policy_name}"
    )


def build_peer_context_by_client(
    *,
    capability_plan: FederatedSslCapabilityPlan,
    ssl_method_config: object | None,
    selected_client_ids: Sequence[str],
    round_index: int,
    client_vectors: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, FederatedSslPeerContext]:
    """round 시작 전 client별 peer/helper context를 만든다.

    script adapter는 capability와 simulation input만 전달한다.
    helper parameter와 nearest-neighbor selection 의미는
    `methods.federated_ssl.hooks.peer_context`가 소유한다.
    """

    policy_name = capability_plan.peer_context_policy_name
    if policy_name == PEER_CONTEXT_NONE:
        return {}
    if policy_name != PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN:
        require_supported_peer_context(capability_plan)
        return {}
    round_state_exchange, effective_parameters = _method_peer_context_sources(
        ssl_method_config
    )
    parameters = resolve_fixed_probe_peer_context_parameters(
        round_state_exchange=round_state_exchange,
        effective_parameters=effective_parameters,
    )
    return build_fixed_probe_peer_context_by_client(
        policy_name=policy_name,
        parameters=parameters,
        selected_client_ids=selected_client_ids,
        round_index=round_index,
        client_vectors=client_vectors,
    )


def _method_peer_context_sources(
    ssl_method_config: object | None,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if ssl_method_config is None:
        raise ValueError(
            "fixed_probe_output_knn peer context requires ssl_method_config "
            "because its parameters come from the method descriptor."
        )
    round_state_exchange = getattr(ssl_method_config, "round_state_exchange", None)
    if not isinstance(round_state_exchange, Mapping):
        raise ValueError(
            "fixed_probe_output_knn peer context requires method "
            "round_state_exchange parameters."
        )
    effective_parameters = getattr(ssl_method_config, "effective_parameters", {})
    if not isinstance(effective_parameters, Mapping):
        effective_parameters = {}
    return round_state_exchange, effective_parameters
