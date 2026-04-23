"""부모용 wellbeing signal 추이 그래프를 제공하는 service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from shared.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalRange,
    WellbeingSignalTimeseriesPayload,
    WellbeingSignalTimeseriesPointPayload,
)

_RANGE_TO_DAYS: dict[WellbeingSignalRange, int] = {
    WellbeingSignalRange.LAST_7_DAYS: 7,
    WellbeingSignalRange.LAST_14_DAYS: 14,
    WellbeingSignalRange.LAST_30_DAYS: 30,
}


@dataclass(slots=True)
class WellbeingTimeseriesService:
    """전체 wellbeing signal 추이를 제공한다.

    MVP 1차 구현은 deterministic mock을 반환한다.
    다음 단계에서는 snapshot repository 기반 조회로 교체한다.
    """

    def get_timeseries(
        self,
        *,
        requested_range: WellbeingSignalRange,
    ) -> WellbeingSignalTimeseriesPayload:
        days = _RANGE_TO_DAYS[requested_range]
        now = datetime.now(tz=timezone.utc)
        start_score = 34.0
        end_score = 61.0
        step = 0.0 if days <= 1 else (end_score - start_score) / float(days - 1)

        points = tuple(
            WellbeingSignalTimeseriesPointPayload(
                ts=(now - timedelta(days=days - index - 1)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                ),
                signal_score=round(start_score + step * index, 2),
            )
            for index in range(days)
        )
        return WellbeingSignalTimeseriesPayload(
            computed_at=now,
            range=requested_range,
            points=points,
        )
