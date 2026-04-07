"""main_server가 소유하는 FL runtime wiring config."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

ROUND_ADAPTER_FAMILY_ENV = "TRACEMIND_ROUND_ADAPTER_FAMILY"
ROUND_AGGREGATION_BACKEND_ENV = "TRACEMIND_ROUND_AGGREGATION_BACKEND"


@dataclass(slots=True)
class ServerRoundRuntimeConfig:
    """서버가 round orchestration을 조립할 때 사용하는 전략 선택 축."""

    adapter_family_name: str = "diagonal_scale"
    aggregation_backend_name: str = "fedavg"


def load_server_round_runtime_config_from_env(
    *,
    environ: Mapping[str, str] | None = None,
) -> ServerRoundRuntimeConfig:
    """환경변수에서 server-owned round runtime config를 읽는다."""

    source = environ or os.environ
    return ServerRoundRuntimeConfig(
        adapter_family_name=source.get(
            ROUND_ADAPTER_FAMILY_ENV,
            "diagonal_scale",
        ),
        aggregation_backend_name=source.get(
            ROUND_AGGREGATION_BACKEND_ENV,
            "fedavg",
        ),
    )
