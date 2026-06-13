"""wellbeing range 계약을 조회 window로 변환한다."""

from __future__ import annotations

from datetime import datetime, timedelta

from agent.src.contracts.wellbeing_signal_contracts import WellbeingSignalRange

_RANGE_TO_DAYS: dict[WellbeingSignalRange, int] = {
    WellbeingSignalRange.LAST_1_DAY: 1,
    WellbeingSignalRange.LAST_7_DAYS: 7,
    WellbeingSignalRange.LAST_14_DAYS: 14,
    WellbeingSignalRange.LAST_30_DAYS: 30,
}


def days_for_range(requested_range: WellbeingSignalRange) -> int:
    return _RANGE_TO_DAYS[requested_range]


def cutoff_for_range(
    anchor: datetime, requested_range: WellbeingSignalRange
) -> datetime:
    days = days_for_range(requested_range)
    return anchor - timedelta(days=max(days - 1, 1))
