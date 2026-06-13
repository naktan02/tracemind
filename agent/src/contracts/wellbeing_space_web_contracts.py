"""가족용 확장 프로그램이 읽는 wellbeing space-web 출력 계약."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalLevel,
    WellbeingSignalRange,
    WellbeingSignalTrend,
)

WELLBEING_SPACE_WEB_V1 = "wellbeing_space_web.v1"

WellbeingSpaceWebSchemaVersion: TypeAlias = Literal["wellbeing_space_web.v1"]
WellbeingSpaceWebNodeId: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, max_length=80),
]
WellbeingSpaceWebStrategyName: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9_.:-]+$"),
]


class WellbeingSpaceWebRelationType(StrEnum):
    """space-web edge가 표현하는 관계 계산 방식."""

    COACTIVATION = "coactivation"
    TRANSITION = "transition"


class WellbeingSpaceWebNodePayload(BaseModel):
    """사용자 공간웹의 단일 카테고리 노드."""

    model_config = ConfigDict(extra="forbid")

    id: WellbeingSpaceWebNodeId
    label: str = Field(min_length=1, max_length=80)
    intensity: float = Field(ge=0.0, le=100.0)
    level: WellbeingSignalLevel
    trend: WellbeingSignalTrend
    observed_event_count: int = Field(ge=0)


class WellbeingSpaceWebEdgePayload(BaseModel):
    """사용자 공간웹의 카테고리 간 관계 edge."""

    model_config = ConfigDict(extra="forbid")

    source: WellbeingSpaceWebNodeId
    target: WellbeingSpaceWebNodeId
    weight: float = Field(ge=0.0, le=100.0)
    relation_type: WellbeingSpaceWebRelationType
    evidence_count: int = Field(ge=0)


class WellbeingSpaceWebPayload(BaseModel):
    """아이용 분석 화면의 space-web graph source of truth."""

    model_config = ConfigDict(extra="forbid")

    schema_version: WellbeingSpaceWebSchemaVersion = WELLBEING_SPACE_WEB_V1
    computed_at: datetime
    range: WellbeingSignalRange
    strategy_name: WellbeingSpaceWebStrategyName
    strategy_version: str = Field(min_length=1, max_length=40)
    nodes: tuple[WellbeingSpaceWebNodePayload, ...] = Field(default_factory=tuple)
    edges: tuple[WellbeingSpaceWebEdgePayload, ...] = Field(default_factory=tuple)
    low_data: bool = False


def load_wellbeing_space_web_payload(path: Path) -> WellbeingSpaceWebPayload:
    """JSON 파일에서 wellbeing space-web payload를 읽는다."""
    return WellbeingSpaceWebPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_wellbeing_space_web_payload(
    path: Path,
    payload: WellbeingSpaceWebPayload,
) -> None:
    """wellbeing space-web payload를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
