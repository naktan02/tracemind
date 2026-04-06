"""로컬 개인화 상태 payload와 직렬화 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

PERSONALIZATION_STATE_V1 = "personalization_state.v1"
PersonalizationStateSchemaVersion: TypeAlias = Literal["personalization_state.v1"]


class PersonalizationWarmupStatus(StrEnum):
    """개인화 상태 warm-up 단계."""

    COLD_START = "cold_start"
    WARMING_UP = "warming_up"
    READY = "ready"


class PersonalizationStatePayload(BaseModel):
    """로컬 persistence에 쓰는 personalization state payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: PersonalizationStateSchemaVersion = PERSONALIZATION_STATE_V1
    state_version: str
    baseline_by_category: dict[str, float] = Field(default_factory=dict)
    threshold_by_category: dict[str, float] = Field(default_factory=dict)
    warmup_status: PersonalizationWarmupStatus
    updated_at: datetime | None = None
    personal_prototype_refs: dict[str, str] = Field(default_factory=dict)
    persistence_features: dict[str, float] = Field(default_factory=dict)
    calibration_notes: str | None = None


def load_personalization_state_payload(path: Path) -> PersonalizationStatePayload:
    """JSON 파일에서 personalization state payload를 읽는다."""
    return PersonalizationStatePayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_personalization_state_payload(
    path: Path,
    payload: PersonalizationStatePayload,
) -> None:
    """personalization state payload를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


PersonalizationState = PersonalizationStatePayload


__all__ = [
    "PERSONALIZATION_STATE_V1",
    "PersonalizationState",
    "PersonalizationStatePayload",
    "PersonalizationStateSchemaVersion",
    "PersonalizationWarmupStatus",
    "dump_personalization_state_payload",
    "load_personalization_state_payload",
]
