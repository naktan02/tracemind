"""PrototypePack 배포용 전송 contract와 직렬화 유틸리티."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOTYPE_PACK_V1 = "prototype_pack.v1"
PrototypePackSchemaVersion: TypeAlias = Literal["prototype_pack.v1"]


class CategoryPrototypePayload(BaseModel):
    """카테고리 하나에 대한 배포용 prototype payload."""

    model_config = ConfigDict(extra="forbid")

    prototype_id: str | None = None
    centroid: list[float]
    sample_count: int = Field(ge=1)


class PrototypePackPayload(BaseModel):
    """agent가 scoring runtime에서 직접 사용하는 prototype pack payload.

    `categories`는 category마다 하나 이상의 prototype을 담는다.
    legacy single-centroid JSON도 읽을 수 있도록 자동 정규화한다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: PrototypePackSchemaVersion = PROTOTYPE_PACK_V1
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
    categories: dict[str, list[CategoryPrototypePayload]]

    @field_validator("categories", mode="before")
    @classmethod
    def _normalize_categories(
        cls,
        value: object,
    ) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise TypeError("categories must be a mapping.")

        normalized: dict[str, object] = {}
        for category, raw_prototypes in value.items():
            if isinstance(raw_prototypes, Mapping):
                normalized[str(category)] = [raw_prototypes]
                continue
            normalized[str(category)] = raw_prototypes
        return normalized


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
    return PrototypePackActivationPointer.model_validate_json(
        path.read_text(encoding="utf-8")
    )


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
    """single prototype pack을 centroid 매핑으로 변환한다.

    multi-prototype category가 있으면 명시적으로 실패시켜 잘못된 단순화를 막는다.
    """
    centroids: dict[str, list[float]] = {}
    multi_categories: list[str] = []
    for category, prototypes in payload.categories.items():
        if len(prototypes) != 1:
            multi_categories.append(category)
            continue
        centroids[category] = list(prototypes[0].centroid)
    if multi_categories:
        raise ValueError(
            "extract_category_centroids only supports single-prototype categories. "
            f"Multi-prototype categories: {sorted(multi_categories)}"
        )
    return centroids


def extract_category_prototypes(
    payload: PrototypePackPayload,
) -> dict[str, tuple[list[float], ...]]:
    """scoring runtime이 바로 쓸 category -> prototype vectors 매핑으로 변환한다."""
    return {
        category: tuple(list(prototype.centroid) for prototype in prototypes)
        for category, prototypes in payload.categories.items()
    }
