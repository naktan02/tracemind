"""PrototypePack 배포용 전송 contract와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CategoryPrototypePayload(BaseModel):
    """카테고리 하나에 대한 배포용 centroid payload."""

    model_config = ConfigDict(extra="forbid")

    centroid: list[float]
    sample_count: int = Field(ge=1)


class PrototypePackPayload(BaseModel):
    """agent가 scoring runtime에서 직접 사용하는 prototype pack payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    prototype_version: str
    embedding_model_id: str
    embedding_model_revision: str
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    mapping_version: str
    build_method: str
    distance_metric: str
    built_at: datetime
    categories: dict[str, CategoryPrototypePayload]


class PrototypePackActivationPointer(BaseModel):
    """현재 활성 prototype pack 버전을 가리키는 포인터."""

    model_config = ConfigDict(extra="forbid")

    prototype_version: str
    activated_at: datetime


class CurrentPrototypePackResponse(BaseModel):
    """중앙 서버가 agent에 내려주는 현재 활성 prototype pack 응답."""

    model_config = ConfigDict(extra="forbid")

    active: PrototypePackActivationPointer
    pack: PrototypePackPayload


class PrototypePackActivationRequest(BaseModel):
    """중앙 서버에서 활성 prototype pack을 바꿀 때 쓰는 요청."""

    model_config = ConfigDict(extra="forbid")

    prototype_version: str


def load_prototype_pack_payload(path: Path) -> PrototypePackPayload:
    """JSON 파일에서 prototype pack payload를 읽는다."""
    return PrototypePackPayload.model_validate_json(path.read_text(encoding="utf-8"))


def dump_prototype_pack_payload(path: Path, payload: PrototypePackPayload) -> None:
    """prototype pack payload를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_activation_pointer(path: Path) -> PrototypePackActivationPointer:
    """활성 버전 포인터 JSON을 읽는다."""
    return PrototypePackActivationPointer.model_validate_json(path.read_text(encoding="utf-8"))


def dump_activation_pointer(
    path: Path,
    pointer: PrototypePackActivationPointer,
) -> None:
    """활성 버전 포인터 JSON을 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pointer.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def extract_category_centroids(
    payload: PrototypePackPayload,
) -> dict[str, list[float]]:
    """scoring runtime이 바로 쓸 centroid 매핑으로 변환한다."""
    return {
        category: prototype.centroid
        for category, prototype in payload.categories.items()
    }
