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

    LAST_7_DAYS = "7d"
    LAST_14_DAYS = "14d"
    LAST_30_DAYS = "30d"


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
