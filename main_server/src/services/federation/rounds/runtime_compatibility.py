"""server-owned round runtime 조합 호환성 검증."""

from __future__ import annotations

from dataclasses import dataclass

from main_server.src.services.federation.rounds.adapter_family_service import (
    build_shared_adapter_round_family,
)
from main_server.src.services.federation.rounds.runtime_config import ServerRoundRuntimeConfig


@dataclass(slots=True)
class ServerRoundRuntimeCompatibility:
    """검증된 server-owned round runtime 조합 요약."""

    adapter_family_name: str
    adapter_kind: str
    aggregation_backend_name: str


def validate_server_round_runtime_config(
    config: ServerRoundRuntimeConfig,
) -> ServerRoundRuntimeCompatibility:
    """서버 round runtime config가 일관된 family/backend 조합인지 검증한다."""

    adapter_family = build_shared_adapter_round_family(
        config.adapter_family_name,
        aggregation_backend_name=config.aggregation_backend_name,
        aggregation_backend_overrides=config.aggregation_backend_overrides,
    )
    if adapter_family.aggregation_backend.adapter_kind != adapter_family.adapter_kind:
        raise ValueError(
            "Incompatible round runtime config: aggregation backend "
            f"{config.aggregation_backend_name} resolved to adapter_kind="
            f"{adapter_family.aggregation_backend.adapter_kind}, "
            f"but family {config.adapter_family_name} expects "
            f"{adapter_family.adapter_kind}."
        )
    return ServerRoundRuntimeCompatibility(
        adapter_family_name=config.adapter_family_name,
        adapter_kind=adapter_family.adapter_kind,
        aggregation_backend_name=config.aggregation_backend_name,
    )
