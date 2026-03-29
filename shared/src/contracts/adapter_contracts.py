"""Vector adapter 상태/업데이트 payload와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class VectorAdapterStatePayload(BaseModel):
    """전역 adapter 상태 payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_id: str
    model_revision: str
    training_scope: str
    dimension_scales: list[float]
    updated_at: datetime


class VectorAdapterDeltaPayload(BaseModel):
    """로컬 학습이 생성한 adapter delta payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_id: str
    base_model_revision: str
    training_scope: str
    dimension_deltas: list[float]
    example_count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime | None = None
    mean_margin: float | None = None
    label_counts: dict[str, int] = Field(default_factory=dict)


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_vector_adapter_state_payload(path: Path) -> VectorAdapterStatePayload:
    """JSON 파일에서 adapter state payload를 읽는다."""
    return VectorAdapterStatePayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_vector_adapter_state_payload(
    path: Path,
    payload: VectorAdapterStatePayload,
) -> None:
    """adapter state payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_vector_adapter_delta_payload(path: Path) -> VectorAdapterDeltaPayload:
    """JSON 파일에서 adapter delta payload를 읽는다."""
    return VectorAdapterDeltaPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_vector_adapter_delta_payload(
    path: Path,
    payload: VectorAdapterDeltaPayload,
) -> None:
    """adapter delta payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)
