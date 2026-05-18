"""FL simulation client update 제출/요약 helper."""

from __future__ import annotations

import json
from typing import Any

from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingUpdateEnvelope,
)


def accept_client_update(
    *,
    server_runtime: SimulationServerRuntime,
    round_id: str,
    update_envelope: TrainingUpdateEnvelope | None,
    update_payload: Any | None,
) -> bool:
    """client update envelope/payload를 server runtime에 제출한다."""

    if update_envelope is None:
        return False
    if update_payload is None:
        raise ValueError("Update envelope exists without update payload.")
    server_runtime.accept_update(
        round_id,
        update_envelope,
        update_payload,
    )
    return True


def payload_byte_count(update_payload: Any | None) -> int | None:
    """report용 client payload JSON byte 수를 계산한다."""

    if update_payload is None:
        return None
    if hasattr(update_payload, "model_dump_json"):
        return len(update_payload.model_dump_json().encode("utf-8"))
    if hasattr(update_payload, "model_dump"):
        update_payload = update_payload.model_dump(mode="json")
    return len(
        json.dumps(
            update_payload,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    )


def extract_delta_l2_norm(
    update_envelope: TrainingUpdateEnvelope | None,
) -> float | None:
    """update envelope metric에서 delta L2 norm을 읽는다."""

    if update_envelope is None:
        return None
    value = update_envelope.client_metrics.get(ClientMetricKeys.DELTA_L2_NORM)
    if value is None:
        return None
    return float(value)


def extract_aggregation_example_count(
    update_envelope: TrainingUpdateEnvelope | None,
) -> int | None:
    """aggregation에 반영될 example count를 읽는다."""

    if update_envelope is None:
        return None
    return int(update_envelope.example_count)
