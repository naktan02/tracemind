"""server-owned round runtime 조합 호환성 검증."""

from __future__ import annotations

from dataclasses import dataclass

from main_server.src.services.federation.rounds.round_state_exchange.executor import (
    validate_default_round_state_exchange_descriptor,
)
from main_server.src.services.federation.rounds.runtime.config import (
    ServerRoundRuntimeConfig,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor

from ..payload_adapters.registry import (
    build_shared_adapter_round_payload_adapter,
)


@dataclass(slots=True)
class ServerRoundRuntimeCompatibility:
    """검증된 server-owned round runtime 조합 요약."""

    payload_adapter_kind: str
    update_family_name: str
    adapter_kind: str
    aggregation_backend_name: str
    method_descriptor_name: str | None = None


def validate_server_round_runtime_config(
    config: ServerRoundRuntimeConfig,
) -> ServerRoundRuntimeCompatibility:
    """서버 round runtime config가 일관된 payload adapter/backend 조합인지 검증한다."""

    payload_adapter = build_shared_adapter_round_payload_adapter(
        config.payload_adapter_kind,
        aggregation_backend_name=config.aggregation_backend_name,
        aggregation_backend_overrides=config.aggregation_backend_overrides,
    )
    if payload_adapter.aggregation_backend.adapter_kind != payload_adapter.adapter_kind:
        raise ValueError(
            "Incompatible round runtime config: aggregation backend "
            f"{config.aggregation_backend_name} resolved to adapter_kind="
            f"{payload_adapter.aggregation_backend.adapter_kind}, "
            f"but payload adapter {config.payload_adapter_kind} expects "
            f"{payload_adapter.adapter_kind}."
        )
    if config.method_descriptor_name is not None:
        method_descriptor = resolve_federated_ssl_method_descriptor(
            config.method_descriptor_name
        )
        if not method_descriptor.runtime_capabilities.live_server_supported:
            raise ValueError(
                "Incompatible round runtime config: method_descriptor="
                f"{method_descriptor.name} does not support live server runtime."
            )
        if method_descriptor.runtime_capabilities.requires_custom_server_runtime:
            raise ValueError(
                "Incompatible round runtime config: method_descriptor="
                f"{method_descriptor.name} requires a custom server runtime "
                "capability that is not wired in the default live server runtime."
            )
        validate_default_round_state_exchange_descriptor(method_descriptor)
        recipe = method_descriptor.recipe
        if recipe is not None and not recipe.supports_runtime_pair(
            update_family_name=config.update_family_name,
            aggregation_backend_name=config.aggregation_backend_name,
        ):
            raise ValueError(
                "Incompatible round runtime config: method recipe does not support "
                "round runtime pair: "
                f"method={method_descriptor.name}, "
                f"update_family={config.update_family_name}, "
                f"aggregation_backend={config.aggregation_backend_name}."
            )
    return ServerRoundRuntimeCompatibility(
        payload_adapter_kind=config.payload_adapter_kind,
        update_family_name=config.update_family_name,
        adapter_kind=payload_adapter.adapter_kind,
        aggregation_backend_name=config.aggregation_backend_name,
        method_descriptor_name=config.method_descriptor_name,
    )
