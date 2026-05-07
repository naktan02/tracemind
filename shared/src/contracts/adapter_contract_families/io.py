"""Shared adapter payload IO helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .base import SharedAdapterStatePayload, SharedAdapterUpdatePayload
from .diagonal_scale import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    VectorAdapterDeltaPayload,
    VectorAdapterStatePayload,
)
from .registry import (
    parse_shared_adapter_state_payload,
    parse_shared_adapter_update_payload,
)


def dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_payload_data(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_shared_adapter_state_payload(path: Path) -> SharedAdapterStatePayload:
    return parse_shared_adapter_state_payload(load_payload_data(path))


def dump_shared_adapter_state_payload(
    path: Path,
    payload: SharedAdapterStatePayload,
) -> None:
    dump_payload(path, payload)


def load_shared_adapter_update_payload(path: Path) -> SharedAdapterUpdatePayload:
    return parse_shared_adapter_update_payload(load_payload_data(path))


def dump_shared_adapter_update_payload(
    path: Path,
    payload: SharedAdapterUpdatePayload,
) -> None:
    dump_payload(path, payload)


def load_vector_adapter_state_payload(path: Path) -> VectorAdapterStatePayload:
    payload = load_shared_adapter_state_payload(path)
    if not isinstance(payload, DiagonalScaleAdapterStatePayload):
        raise ValueError(
            "Expected diagonal_scale adapter state payload, "
            f"got {payload.adapter_kind}."
        )
    return payload


def dump_vector_adapter_state_payload(
    path: Path,
    payload: VectorAdapterStatePayload,
) -> None:
    dump_shared_adapter_state_payload(path, payload)


def load_vector_adapter_delta_payload(path: Path) -> VectorAdapterDeltaPayload:
    payload = load_shared_adapter_update_payload(path)
    if not isinstance(payload, DiagonalScaleAdapterUpdatePayload):
        raise ValueError(
            "Expected diagonal_scale adapter update payload, "
            f"got {payload.adapter_kind}."
        )
    return payload


def dump_vector_adapter_delta_payload(
    path: Path,
    payload: VectorAdapterDeltaPayload,
) -> None:
    dump_shared_adapter_update_payload(path, payload)
