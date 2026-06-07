"""부모용 wellbeing signal 추이 그래프를 제공하는 service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from agent.src.infrastructure.repositories.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.services.wellbeing.projection_service import (
    WellbeingSignalProjectionService,
)
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

    snapshot repository를 source of truth로 사용한다. 저장된 구간이 없으면
    빈 point 묶음으로 아직 관측된 추이가 없음을 표현한다.
    """

    repository: WellbeingSnapshotRepository | None = None
    projection_service: WellbeingSignalProjectionService | None = None

    def get_timeseries(
        self,
        *,
        requested_range: WellbeingSignalRange,
    ) -> WellbeingSignalTimeseriesPayload:
        if self.projection_service is not None:
            self.projection_service.refresh_from_runtime()
        if self.repository is not None:
            now = datetime.now(tz=timezone.utc)
            days = _RANGE_TO_DAYS[requested_range]
            summaries = self.repository.list_summaries_since(
                cutoff=now - timedelta(days=days - 1),
            )
            if not summaries:
                latest_summary = self.repository.load_latest_summary()
                if latest_summary is not None:
                    summaries = self.repository.list_summaries_since(
                        cutoff=latest_summary.computed_at - timedelta(days=days - 1),
                    )
            if summaries:
                return WellbeingSignalTimeseriesPayload(
                    computed_at=now,
                    range=requested_range,
                    points=tuple(
                        WellbeingSignalTimeseriesPointPayload(
                            ts=summary.computed_at,
                            signal_score=summary.signal_score,
                        )
                        for summary in summaries
                    ),
                )

        now = datetime.now(tz=timezone.utc)
        return WellbeingSignalTimeseriesPayload(
            computed_at=now,
            range=requested_range,
            points=(),
        )
