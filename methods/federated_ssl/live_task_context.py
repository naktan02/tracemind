"""Live TrainingTask FSSL context 해석 helper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from methods.federated_ssl.hooks.peer_context import FederatedSslPeerContext
from methods.federated_ssl.method_config_surface import DEFAULT_LOCAL_BUDGET_POLICY


def build_method_config_from_live_fssl_context(
    *,
    fssl_method: str | None,
    fssl_context: Mapping[str, object] | None,
) -> dict[str, object]:
    """live task context를 method parameter snapshot 입력으로 정규화한다."""

    method_name = _optional_name(fssl_method)
    if method_name is None:
        raise ValueError("fssl_method is required for method-owned local runtime.")
    context = {} if fssl_context is None else dict(fssl_context)
    method_config = {
        "name": method_name,
        "use_original_parameters": True,
        "parameter_overrides": {},
        "local_budget_policy": DEFAULT_LOCAL_BUDGET_POLICY,
    }
    raw_method_config = context.get("method_config")
    if isinstance(raw_method_config, Mapping):
        method_config.update(dict(raw_method_config))
        method_config["name"] = method_name
    raw_peer_context = context.get("peer_context")
    if isinstance(raw_peer_context, Mapping):
        scenario = _optional_name(raw_peer_context.get("scenario"))
        if scenario is not None:
            method_config["scenario"] = scenario
    return method_config


def build_peer_context_from_live_fssl_context(
    *,
    fssl_context: Mapping[str, object] | None,
    client_id: str,
    default_policy_name: str | None = None,
) -> FederatedSslPeerContext | None:
    """live task의 peer_context payload를 method-owned peer context object로 바꾼다."""

    context = {} if fssl_context is None else dict(fssl_context)
    raw_peer_context = context.get("peer_context")
    if not isinstance(raw_peer_context, Mapping):
        return None
    policy_name = _optional_name(raw_peer_context.get("policy_name"))
    if policy_name is None:
        policy_name = _optional_name(default_policy_name)
    if policy_name is None:
        return None
    client_payload = _find_peer_context_client_payload(
        raw_peer_context=raw_peer_context,
        client_id=client_id,
    )
    return FederatedSslPeerContext(
        client_id=client_id,
        policy_name=policy_name,
        round_index_zero_based=_round_index_zero_based(raw_peer_context),
        helper_client_ids=(
            ()
            if client_payload is None
            else tuple(
                str(helper_id)
                for helper_id in client_payload.get("helper_client_ids", ())
            )
        ),
        refreshed=not bool(raw_peer_context.get("warmup", False)),
        metadata={
            "source_round_id": raw_peer_context.get("source_round_id"),
            "context_kind": context.get("context_kind"),
            "method_name": context.get("method_name"),
            "summary_metrics": dict(raw_peer_context.get("summary_metrics", {})),
        },
    )


def _find_peer_context_client_payload(
    *,
    raw_peer_context: Mapping[str, object],
    client_id: str,
) -> Mapping[str, object] | None:
    raw_client_contexts = raw_peer_context.get("client_contexts", ())
    if not isinstance(raw_client_contexts, Sequence) or isinstance(
        raw_client_contexts,
        (str, bytes),
    ):
        return None
    for item in raw_client_contexts:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("client_id", "")).strip() == client_id:
            return item
    return None


def _round_index_zero_based(raw_peer_context: Mapping[str, object]) -> int:
    raw_round_index = raw_peer_context.get("round_index_zero_based")
    if raw_round_index is None:
        return 0
    return max(0, int(raw_round_index))


def _optional_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
