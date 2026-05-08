"""main_server가 소유하는 FL runtime wiring config."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
)

from ..aggregation.models import AggregationConfigScalar

ROUND_ADAPTER_FAMILY_ENV = "TRACEMIND_ROUND_ADAPTER_FAMILY"
ROUND_AGGREGATION_BACKEND_ENV = "TRACEMIND_ROUND_AGGREGATION_BACKEND"
ROUND_AGGREGATION_BACKEND_CONFIG_ENV = "TRACEMIND_ROUND_AGGREGATION_BACKEND_CONFIG"
ROUND_METHOD_DESCRIPTOR_ENV = "TRACEMIND_ROUND_METHOD_DESCRIPTOR"
DEFAULT_SERVER_ROUND_ADAPTER_FAMILY_NAME = DIAGONAL_SCALE_ADAPTER_KIND


@dataclass(slots=True)
class ServerRoundRuntimeConfig:
    """서버가 round orchestration을 조립할 때 사용하는 전략 선택 축."""

    adapter_family_name: str = DEFAULT_SERVER_ROUND_ADAPTER_FAMILY_NAME
    aggregation_backend_name: str = "fedavg"
    method_descriptor_name: str | None = None
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] = field(
        default_factory=dict
    )


def load_server_round_runtime_config_from_env(
    *,
    environ: Mapping[str, str] | None = None,
) -> ServerRoundRuntimeConfig:
    """환경변수에서 server-owned round runtime config를 읽는다."""

    source = environ or os.environ
    return ServerRoundRuntimeConfig(
        adapter_family_name=source.get(
            ROUND_ADAPTER_FAMILY_ENV,
            DEFAULT_SERVER_ROUND_ADAPTER_FAMILY_NAME,
        ),
        aggregation_backend_name=source.get(
            ROUND_AGGREGATION_BACKEND_ENV,
            "fedavg",
        ),
        method_descriptor_name=_optional_env_value(source, ROUND_METHOD_DESCRIPTOR_ENV),
        aggregation_backend_overrides=_load_aggregation_backend_overrides(source),
    )


def _load_aggregation_backend_overrides(
    source: Mapping[str, str],
) -> Mapping[str, AggregationConfigScalar]:
    raw_value = source.get(ROUND_AGGREGATION_BACKEND_CONFIG_ENV)
    if raw_value is None or not raw_value.strip():
        return {}
    payload = json.loads(raw_value)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{ROUND_AGGREGATION_BACKEND_CONFIG_ENV} must be a JSON object."
        )
    overrides: dict[str, AggregationConfigScalar] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{ROUND_AGGREGATION_BACKEND_CONFIG_ENV} keys must be strings."
            )
        if isinstance(value, (str, int, float, bool)):
            overrides[key] = value
            continue
        raise ValueError(
            f"{ROUND_AGGREGATION_BACKEND_CONFIG_ENV} values must be scalar JSON "
            f"types, got {type(value)!r} for key={key!r}."
        )
    return overrides


def _optional_env_value(source: Mapping[str, str], key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
