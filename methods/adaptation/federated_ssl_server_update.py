"""Payload adapter kind별 FL SSL server update policy backend resolver."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol

from methods.adaptation.payload_adapter_modules import (
    normalize_payload_adapter_kind,
    payload_adapter_module_name,
)
from methods.federated_ssl.capability_axes import SERVER_UPDATE_FEDAVG_MERGED_DELTA


class FederatedSslServerUpdateBackendResolver(Protocol):
    """server_update_policy를 family별 aggregation backend로 해석하는 함수."""

    def __call__(
        self,
        *,
        server_update_policy_name: str,
        aggregation_backend_name: str,
    ) -> str:
        """effective aggregation backend 이름을 반환한다."""


_SERVER_UPDATE_BACKEND_RESOLVERS: dict[
    str,
    FederatedSslServerUpdateBackendResolver,
] = {}


def register_federated_ssl_server_update_backend_resolver(
    adapter_kind: str,
) -> Callable[
    [FederatedSslServerUpdateBackendResolver],
    FederatedSslServerUpdateBackendResolver,
]:
    """payload adapter 구현 owner 옆에서 FL SSL server update resolver를 등록한다."""

    normalized_adapter_kind = _normalize_payload_adapter_kind(adapter_kind)

    def _decorator(
        resolver: FederatedSslServerUpdateBackendResolver,
    ) -> FederatedSslServerUpdateBackendResolver:
        if normalized_adapter_kind in _SERVER_UPDATE_BACKEND_RESOLVERS:
            raise ValueError(
                "Duplicate FL SSL server update backend resolver registration: "
                f"{normalized_adapter_kind}"
            )
        _SERVER_UPDATE_BACKEND_RESOLVERS[normalized_adapter_kind] = resolver
        return resolver

    return _decorator


def resolve_federated_ssl_server_update_backend_name(
    *,
    payload_adapter_kind: str,
    server_update_policy_name: str | None,
    aggregation_backend_name: str,
) -> str:
    """server_update_policy가 요구하는 effective aggregation backend를 반환한다."""

    normalized_policy = (
        SERVER_UPDATE_FEDAVG_MERGED_DELTA
        if server_update_policy_name is None
        else server_update_policy_name.strip().lower().replace("-", "_")
    )
    if normalized_policy == SERVER_UPDATE_FEDAVG_MERGED_DELTA:
        return aggregation_backend_name

    normalized_adapter_kind = _normalize_payload_adapter_kind(payload_adapter_kind)
    resolver = _SERVER_UPDATE_BACKEND_RESOLVERS.get(normalized_adapter_kind)
    if resolver is None:
        _import_federated_ssl_module_for_adapter_kind(normalized_adapter_kind)
        resolver = _SERVER_UPDATE_BACKEND_RESOLVERS.get(normalized_adapter_kind)
    if resolver is None:
        raise ValueError(
            "server_update_policy is not supported by payload adapter: "
            f"{normalized_policy} with {normalized_adapter_kind}."
        )
    return resolver(
        server_update_policy_name=normalized_policy,
        aggregation_backend_name=aggregation_backend_name,
    )


def _import_federated_ssl_module_for_adapter_kind(
    normalized_adapter_kind: str,
) -> None:
    module_name = payload_adapter_module_name(
        payload_adapter_kind=normalized_adapter_kind,
        submodule="federated_ssl.server_update_policy",
    )
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return
        raise


def _normalize_payload_adapter_kind(adapter_kind: str) -> str:
    return normalize_payload_adapter_kind(adapter_kind)
