"""부모용 wellbeing signal 추이 그래프를 제공하는 service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalRange,
    WellbeingSignalTimeseriesPayload,
    WellbeingSignalTimeseriesPointPayload,
)
from agent.src.features.wellbeing.range_window import cutoff_for_range
from agent.src.features.wellbeing.signal.projection_service import (
    WellbeingSignalProjectionService,
)
from agent.src.features.wellbeing.storage.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)


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
            summaries = self.repository.list_summaries_since(
                cutoff=cutoff_for_range(now, requested_range),
            )
            if not summaries:
                latest_summary = self.repository.load_latest_summary()
                if latest_summary is not None:
                    summaries = self.repository.list_summaries_since(
                        cutoff=cutoff_for_range(
                            latest_summary.computed_at,
                            requested_range,
                        ),
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
