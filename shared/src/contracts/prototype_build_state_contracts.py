"""Single-centroid prototype build-state contract와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

PROTOTYPE_BUILD_STATE_V1 = "prototype_build_state.v1"
PrototypeBuildStateSchemaVersion: TypeAlias = Literal["prototype_build_state.v1"]


class CategoryPrototypeBuildStatePayload(BaseModel):
    """카테고리 하나에 대한 build-state 누적값."""

    model_config = ConfigDict(extra="forbid")

    embedding_sum: list[float]
    sample_count: int = Field(ge=1)


class SingleCategoryPrototypeBuildStatePayload(CategoryPrototypeBuildStatePayload):
    """single mean-centroid build-state용 category 누적값."""


class SinglePrototypeBuildStatePayload(BaseModel):
    """single mean-centroid exact incremental update용 build-state payload.

    현재 v1 payload는 category별 embedding 합과 sample 수만 담는다.
    따라서 exact incremental merge는 single mean-centroid builder에만 해당한다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: PrototypeBuildStateSchemaVersion = PROTOTYPE_BUILD_STATE_V1
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
    categories: dict[str, SingleCategoryPrototypeBuildStatePayload]


# Backward-compatible aliases. New code should prefer explicit single-* names.
PrototypeBuildStatePayload = SinglePrototypeBuildStatePayload
CategoryPrototypeBuildStatePayload = SingleCategoryPrototypeBuildStatePayload


def load_prototype_build_state_payload(path: Path) -> SinglePrototypeBuildStatePayload:
    """JSON 파일에서 prototype build state payload를 읽는다."""
    return SinglePrototypeBuildStatePayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_prototype_build_state_payload(
    path: Path,
    payload: SinglePrototypeBuildStatePayload,
) -> None:
    """prototype build state payload를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
