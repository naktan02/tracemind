"""Wellbeing signal contract unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent.src.contracts.wellbeing_signal_contracts import (
    ParentUnlockRequestPayload,
    ParentUnlockResponsePayload,
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalRange,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTimeseriesPayload,
    WellbeingSignalTrend,
    dump_wellbeing_signal_summary_payload,
    load_wellbeing_signal_summary_payload,
)


def test_wellbeing_signal_summary_payload_accepts_canonical_fields() -> None:
    payload = WellbeingSignalSummaryPayload(
        computed_at=datetime(2026, 4, 24, 10, 30, tzinfo=timezone.utc),
        signal_score=72.0,
        signal_level=WellbeingSignalLevel.HIGH,
        signal_label="주의 필요",
        trend=WellbeingSignalTrend.RISING,
        summary="최근 상태가 평소보다 높게 관찰되었습니다.",
        action_tip="오늘은 짧게 대화를 시도해 보세요.",
        confidence=WellbeingSignalConfidence.MEDIUM,
        low_data=False,
    )

    assert payload.schema_version == "wellbeing_signal_summary.v1"
    assert payload.signal_level == WellbeingSignalLevel.HIGH
    assert payload.trend == WellbeingSignalTrend.RISING


def test_wellbeing_signal_summary_payload_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        WellbeingSignalSummaryPayload(
            computed_at=datetime(2026, 4, 24, 10, 30, tzinfo=timezone.utc),
            signal_score=120.0,
            signal_level=WellbeingSignalLevel.VERY_HIGH,
            signal_label="즉시 확인",
            trend=WellbeingSignalTrend.RISING,
            summary="점수가 비정상적으로 높습니다.",
            action_tip="잠시 사용을 멈추고 보호자와 확인하세요.",
            confidence=WellbeingSignalConfidence.HIGH,
        )


def test_wellbeing_signal_timeseries_payload_accepts_points() -> None:
    payload = WellbeingSignalTimeseriesPayload.model_validate(
        {
            "computed_at": "2026-04-24T10:30:00Z",
            "range": "1d",
            "points": [
                {"ts": "2026-04-22T00:00:00Z", "signal_score": 51.0},
                {"ts": "2026-04-23T00:00:00Z", "signal_score": 61.0},
                {"ts": "2026-04-24T00:00:00Z", "signal_score": 72.0},
            ],
        }
    )

    assert payload.range == WellbeingSignalRange.LAST_1_DAY
    assert len(payload.points) == 3
    assert payload.points[-1].signal_score == 72.0


def test_parent_unlock_request_payload_requires_numeric_pin() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        ParentUnlockRequestPayload(pin="12ab")


def test_parent_unlock_response_payload_supports_denied_state() -> None:
    payload = ParentUnlockResponsePayload(
        granted=False,
        remaining_attempts=3,
    )

    assert payload.granted is False
    assert payload.remaining_attempts == 3
    assert payload.session_token is None


def test_wellbeing_signal_summary_dump_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "summary.json"
    payload = WellbeingSignalSummaryPayload(
        computed_at=datetime(2026, 4, 24, 10, 30, tzinfo=timezone.utc),
        signal_score=34.0,
        signal_level=WellbeingSignalLevel.MODERATE,
        signal_label="관찰 필요",
        trend=WellbeingSignalTrend.STEADY,
        summary="최근 상태가 비교적 안정적으로 유지되고 있습니다.",
        action_tip="오늘 저녁에 짧게 안부를 물어보세요.",
        confidence=WellbeingSignalConfidence.MEDIUM,
        low_data=False,
    )

    dump_wellbeing_signal_summary_payload(path, payload)
    loaded = load_wellbeing_signal_summary_payload(path)

    assert loaded == payload
