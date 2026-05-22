"""FL simulation peer context exchange adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from methods.federated_ssl.capability_plan import (
    PEER_CONTEXT_NONE,
    PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK,
    FederatedSslCapabilityPlan,
)
from methods.federated_ssl.peer_context import (
    FederatedSslPeerContext,
    NearestPeerClientIndex,
    should_refresh_peer_context,
)


def require_supported_peer_context(
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """v1 simulation에서 지원되는 peer context인지 확인한다."""

    if capability_plan.peer_context_policy_name in {
        PEER_CONTEXT_NONE,
        PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK,
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

    `prediction_similarity_topk`는 mechanism만 공통으로 구현한다. helper 개수와
    refresh interval 같은 의미 있는 값은 method effective parameters에서 읽고,
    FedMatch 원본 의미에 맞춰 KDTree 우선 nearest-neighbor index를 사용한다.
    """

    policy_name = capability_plan.peer_context_policy_name
    if policy_name == PEER_CONTEXT_NONE:
        return {}
    if policy_name != PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK:
        require_supported_peer_context(capability_plan)
        return {}
    if round_index <= 0:
        raise ValueError("round_index must be one-based and positive.")

    parameters = _resolve_method_peer_context_parameters(ssl_method_config)
    num_helpers = _required_positive_or_zero_int(parameters, "num_helpers")
    refresh_interval = _required_positive_int(parameters, "refresh_interval")
    round_index_zero_based = round_index - 1
    refresh_due = should_refresh_peer_context(
        round_index_zero_based=round_index_zero_based,
        refresh_interval=refresh_interval,
    )
    vectors = {} if client_vectors is None else dict(client_vectors)
    helper_index = (
        NearestPeerClientIndex(client_vectors=vectors, prefer_kdtree=True)
        if refresh_due and vectors
        else None
    )
    contexts: dict[str, FederatedSslPeerContext] = {}
    for client_id in selected_client_ids:
        has_selection_vector = client_id in vectors
        helper_client_ids: tuple[str, ...] = ()
        if refresh_due and has_selection_vector and helper_index is not None:
            helper_client_ids = helper_index.query(
                client_id=client_id,
                peer_count=num_helpers,
            )
        contexts[client_id] = FederatedSslPeerContext(
            client_id=client_id,
            policy_name=policy_name,
            round_index_zero_based=round_index_zero_based,
            helper_client_ids=helper_client_ids,
            refreshed=refresh_due and has_selection_vector,
            metadata={
                "num_helpers": num_helpers,
                "refresh_interval": refresh_interval,
                "refresh_due": refresh_due,
                "has_selection_vector": has_selection_vector,
                "selection_vector_source": (
                    "provided" if has_selection_vector else "unavailable"
                ),
                "selection_index_backend": (
                    helper_index.backend_name if helper_index is not None else "none"
                ),
                "selection_query_size": (
                    helper_index.query_size_including_self(
                        peer_count=num_helpers,
                    )
                    if helper_index is not None
                    else 0
                ),
                "parameter_source": "effective_parameters",
            },
        )
    return contexts


def _resolve_method_peer_context_parameters(
    ssl_method_config: object | None,
) -> Mapping[str, object]:
    if ssl_method_config is None:
        raise ValueError(
            "prediction_similarity_topk peer context requires ssl_method_config "
            "because its parameters come from the method descriptor."
        )
    round_state_exchange = getattr(ssl_method_config, "round_state_exchange", None)
    if not isinstance(round_state_exchange, Mapping):
        raise ValueError(
            "prediction_similarity_topk peer context requires method "
            "round_state_exchange parameters."
        )
    effective_parameters = getattr(ssl_method_config, "effective_parameters", {})
    if not isinstance(effective_parameters, Mapping):
        effective_parameters = {}
    resolved = dict(round_state_exchange)
    if "num_helpers" in effective_parameters:
        resolved["num_helpers"] = effective_parameters["num_helpers"]
    if "helper_refresh_interval" in effective_parameters:
        resolved["refresh_interval"] = effective_parameters["helper_refresh_interval"]
    elif "refresh_interval" in effective_parameters:
        resolved["refresh_interval"] = effective_parameters["refresh_interval"]
    return resolved


def _required_positive_or_zero_int(
    parameters: Mapping[str, object],
    key: str,
) -> int:
    if key not in parameters:
        raise ValueError(f"peer context method parameter is missing: {key}.")
    value = int(parameters[key])
    if value < 0:
        raise ValueError(f"peer context method parameter must be non-negative: {key}.")
    return value


def _required_positive_int(
    parameters: Mapping[str, object],
    key: str,
) -> int:
    value = _required_positive_or_zero_int(parameters, key)
    if value <= 0:
        raise ValueError(f"peer context method parameter must be positive: {key}.")
    return value
