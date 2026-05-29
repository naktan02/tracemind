"""main_server가 소유하는 FL runtime wiring config."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from methods.federated_ssl.runtime_fallbacks import (
    RUNTIME_FALLBACK_SERVER_ROUND_PROFILE,
)

from ..aggregation.models import AggregationConfigScalar

ROUND_PAYLOAD_ADAPTER_KIND_ENV = "TRACEMIND_ROUND_PAYLOAD_ADAPTER_KIND"
ROUND_UPDATE_FAMILY_ENV = "TRACEMIND_ROUND_UPDATE_FAMILY"
ROUND_AGGREGATION_BACKEND_ENV = "TRACEMIND_ROUND_AGGREGATION_BACKEND"
ROUND_AGGREGATION_BACKEND_CONFIG_ENV = "TRACEMIND_ROUND_AGGREGATION_BACKEND_CONFIG"
ROUND_METHOD_DESCRIPTOR_ENV = "TRACEMIND_ROUND_METHOD_DESCRIPTOR"


@dataclass(frozen=True, slots=True)
class ServerRoundRuntimeProfile:
    """명시 runtime config가 없을 때 쓰는 compatibility profile."""

    profile_name: str
    payload_adapter_kind: str
    update_family_name: str
    aggregation_backend_name: str
    method_descriptor_name: str | None = None
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] = field(
        default_factory=dict
    )


DEFAULT_PEFT_CLASSIFIER_SERVER_ROUND_RUNTIME_PROFILE = ServerRoundRuntimeProfile(
    profile_name=RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.profile_name,
    payload_adapter_kind=RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.payload_adapter_kind,
    update_family_name=RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.update_family_name,
    aggregation_backend_name=(
        RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.aggregation_backend_name
    ),
    method_descriptor_name=RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.method_descriptor_name,
    aggregation_backend_overrides=dict(
        RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.aggregation_backend_overrides
    ),
)
DEFAULT_SERVER_ROUND_RUNTIME_PROFILE = (
    DEFAULT_PEFT_CLASSIFIER_SERVER_ROUND_RUNTIME_PROFILE
)


@dataclass(slots=True, init=False)
class ServerRoundRuntimeConfig:
    """서버가 round orchestration을 조립할 때 사용하는 전략 선택 축."""

    payload_adapter_kind: str
    update_family_name: str = DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.update_family_name
    aggregation_backend_name: str = (
        DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.aggregation_backend_name
    )
    method_descriptor_name: str | None = (
        DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.method_descriptor_name
    )
    aggregation_backend_overrides: Mapping[str, AggregationConfigScalar] = field(
        default_factory=dict
    )

    def __init__(
        self,
        *,
        payload_adapter_kind: str | None = None,
        update_family_name: str = (
            DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.update_family_name
        ),
        aggregation_backend_name: str = (
            DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.aggregation_backend_name
        ),
        method_descriptor_name: str | None = (
            DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.method_descriptor_name
        ),
        aggregation_backend_overrides: (
            Mapping[str, AggregationConfigScalar] | None
        ) = None,
    ) -> None:
        self.payload_adapter_kind = (
            _optional_str(payload_adapter_kind)
            or DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.payload_adapter_kind
        )
        self.update_family_name = update_family_name
        self.aggregation_backend_name = aggregation_backend_name
        self.method_descriptor_name = method_descriptor_name
        self.aggregation_backend_overrides = dict(aggregation_backend_overrides or {})


def load_server_round_runtime_config_from_env(
    *,
    environ: Mapping[str, str] | None = None,
) -> ServerRoundRuntimeConfig:
    """환경변수에서 server-owned round runtime config를 읽는다."""

    source = environ or os.environ
    return ServerRoundRuntimeConfig(
        payload_adapter_kind=(
            source.get(ROUND_PAYLOAD_ADAPTER_KIND_ENV)
            or DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.payload_adapter_kind
        ),
        update_family_name=source.get(
            ROUND_UPDATE_FAMILY_ENV,
            DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.update_family_name,
        ),
        aggregation_backend_name=source.get(
            ROUND_AGGREGATION_BACKEND_ENV,
            DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.aggregation_backend_name,
        ),
        method_descriptor_name=(
            _optional_env_value(source, ROUND_METHOD_DESCRIPTOR_ENV)
            or DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.method_descriptor_name
        ),
        aggregation_backend_overrides=(
            _load_aggregation_backend_overrides(source)
            or DEFAULT_SERVER_ROUND_RUNTIME_PROFILE.aggregation_backend_overrides
        ),
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


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
