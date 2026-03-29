"""Prototype build state 전송 contract와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CategoryPrototypeBuildStatePayload(BaseModel):
    """카테고리 하나에 대한 build-state 누적값."""

    model_config = ConfigDict(extra="forbid")

    embedding_sum: list[float]
    sample_count: int = Field(ge=1)


class PrototypeBuildStatePayload(BaseModel):
    """정확한 incremental update를 위한 prototype build state payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    prototype_version: str
    embedding_backend: str
    embedding_model_id: str
    embedding_model_revision: str
    normalize_embeddings: bool = True
    task_prefix: str = ""
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    mapping_version: str
    build_method: str
    distance_metric: str
    built_at: datetime
    categories: dict[str, CategoryPrototypeBuildStatePayload]


def load_prototype_build_state_payload(path: Path) -> PrototypeBuildStatePayload:
    """JSON 파일에서 prototype build state payload를 읽는다."""
    return PrototypeBuildStatePayload.model_validate_json(path.read_text(encoding="utf-8"))


def dump_prototype_build_state_payload(
    path: Path,
    payload: PrototypeBuildStatePayload,
) -> None:
    """prototype build state payload를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
