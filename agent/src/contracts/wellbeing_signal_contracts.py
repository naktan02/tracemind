"""가족용 확장 프로그램이 읽는 wellbeing signal 출력 계약."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

WELLBEING_SIGNAL_SUMMARY_V1 = "wellbeing_signal_summary.v1"
WELLBEING_SIGNAL_TIMESERIES_V1 = "wellbeing_signal_timeseries.v1"
PARENT_UNLOCK_RESPONSE_V1 = "parent_unlock_response.v1"

WellbeingSignalSummarySchemaVersion: TypeAlias = Literal["wellbeing_signal_summary.v1"]
WellbeingSignalTimeseriesSchemaVersion: TypeAlias = Literal[
    "wellbeing_signal_timeseries.v1"
]
ParentUnlockResponseSchemaVersion: TypeAlias = Literal["parent_unlock_response.v1"]

PinCode: TypeAlias = Annotated[str, StringConstraints(pattern=r"^\d{4,6}$")]


class WellbeingSignalLevel(StrEnum):
    """외부 UI가 보는 전체 wellbeing signal 수준."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class WellbeingSignalTrend(StrEnum):
    """최근 wellbeing signal 변화 방향."""

    RISING = "rising"
    STEADY = "steady"
    FALLING = "falling"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class WellbeingSignalConfidence(StrEnum):
    """현재 출력에 대한 신뢰도 수준."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WellbeingSignalRange(StrEnum):
    """부모용 추이 화면이 요청하는 기간."""

    LAST_1_DAY = "1d"
    LAST_7_DAYS = "7d"
    LAST_14_DAYS = "14d"
    LAST_30_DAYS = "30d"


class ParentWellbeingGuidancePayload(BaseModel):
    """부모 화면이 표시하는 규칙 기반 대응 안내."""

    model_config = ConfigDict(extra="forbid")

    response_priority: str = Field(min_length=1)
    conversation_starter: str = Field(min_length=1)
    caution_note: str = Field(min_length=1)


DEFAULT_PARENT_WELLBEING_GUIDANCE = ParentWellbeingGuidancePayload(
    response_priority="짧은 안부 확인부터 시작하세요.",
    conversation_starter=(
        "오늘 하루 중 가장 힘들었던 순간이 있었는지 가볍게 물어보세요."
    ),
    caution_note="원문을 확인하려 하기보다 현재 필요한 도움을 먼저 물어보세요.",
)


class WellbeingSignalSummaryPayload(BaseModel):
    """아이용/부모용 현재 상태 카드 source of truth."""

    model_config = ConfigDict(extra="forbid")

    schema_version: WellbeingSignalSummarySchemaVersion = WELLBEING_SIGNAL_SUMMARY_V1
    computed_at: datetime
    signal_score: float = Field(ge=0.0, le=100.0)
    signal_level: WellbeingSignalLevel
    signal_label: str = Field(min_length=1)
    trend: WellbeingSignalTrend
    summary: str = Field(min_length=1)
    action_tip: str = Field(min_length=1)
    parent_guidance: ParentWellbeingGuidancePayload = Field(
        default_factory=lambda: DEFAULT_PARENT_WELLBEING_GUIDANCE.model_copy()
    )
    confidence: WellbeingSignalConfidence
    low_data: bool = False


class WellbeingSignalTimeseriesPointPayload(BaseModel):
    """추이 그래프의 단일 시점."""

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    signal_score: float = Field(ge=0.0, le=100.0)


class WellbeingSignalTimeseriesPayload(BaseModel):
    """부모용 전체 wellbeing signal 추이 그래프 source of truth."""

    model_config = ConfigDict(extra="forbid")

    schema_version: WellbeingSignalTimeseriesSchemaVersion = (
        WELLBEING_SIGNAL_TIMESERIES_V1
    )
    computed_at: datetime
    range: WellbeingSignalRange
    points: tuple[WellbeingSignalTimeseriesPointPayload, ...] = Field(
        default_factory=tuple
    )


class ParentUnlockRequestPayload(BaseModel):
    """부모용 상세 화면 진입을 위한 PIN 요청."""

    model_config = ConfigDict(extra="forbid")

    pin: PinCode


class ParentUnlockResponsePayload(BaseModel):
    """부모용 상세 화면 접근 결과."""

    model_config = ConfigDict(extra="forbid")

    schema_version: ParentUnlockResponseSchemaVersion = PARENT_UNLOCK_RESPONSE_V1
    granted: bool
    session_token: str | None = None
    session_expires_at: datetime | None = None
    remaining_attempts: int | None = Field(default=None, ge=0)
    locked_until: datetime | None = None


def load_wellbeing_signal_summary_payload(
    path: Path,
) -> WellbeingSignalSummaryPayload:
    """JSON 파일에서 wellbeing signal summary payload를 읽는다."""
    return WellbeingSignalSummaryPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_wellbeing_signal_summary_payload(
    path: Path,
    payload: WellbeingSignalSummaryPayload,
) -> None:
    """wellbeing signal summary payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_wellbeing_signal_timeseries_payload(
    path: Path,
) -> WellbeingSignalTimeseriesPayload:
    """JSON 파일에서 wellbeing signal timeseries payload를 읽는다."""
    return WellbeingSignalTimeseriesPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_wellbeing_signal_timeseries_payload(
    path: Path,
    payload: WellbeingSignalTimeseriesPayload,
) -> None:
    """wellbeing signal timeseries payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
