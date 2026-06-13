"""Analysis event 흐름을 wellbeing space-web payload로 투영한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent.src.contracts.wellbeing_signal_contracts import WellbeingSignalRange
from agent.src.contracts.wellbeing_space_web_contracts import WellbeingSpaceWebPayload
from agent.src.features.inference.interpretation.baseline import BaselineService
from agent.src.features.wellbeing.space_web.coactivation_delta_strategy import (
    CoactivationDeltaSpaceWebStrategy,
)
from agent.src.features.wellbeing.space_web.projection_strategy import (
    SpaceWebProjectionContext,
    SpaceWebProjectionStrategy,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

_RANGE_TO_DAYS: dict[WellbeingSignalRange, int] = {
    WellbeingSignalRange.LAST_7_DAYS: 7,
    WellbeingSignalRange.LAST_14_DAYS: 14,
    WellbeingSignalRange.LAST_30_DAYS: 30,
}


@dataclass(slots=True)
class WellbeingSpaceWebProjectionService:
    """아이용 분석 화면의 category space-web view-model을 만든다."""

    analysis_event_repository: AnalysisEventRepository
    baseline_service: BaselineService = field(default_factory=BaselineService)
    strategy: SpaceWebProjectionStrategy = field(
        default_factory=CoactivationDeltaSpaceWebStrategy
    )
    category_label_overrides: Mapping[str, str] = field(default_factory=dict)
    now_provider: Callable[[], datetime] = field(
        default=lambda: datetime.now(tz=timezone.utc)
    )

    def get_space_web(
        self,
        *,
        requested_range: WellbeingSignalRange,
    ) -> WellbeingSpaceWebPayload:
        computed_at = self.now_provider()
        range_days = _RANGE_TO_DAYS[requested_range]
        history_days = range_days + self.baseline_service.config.lookback_days
        events = sorted(
            self.analysis_event_repository.get_recent(days=history_days),
            key=lambda event: event.occurred_at,
        )
        window_start = computed_at - timedelta(days=range_days - 1)
        recent_events = [
            event
            for event in events
            if window_start <= event.occurred_at <= computed_at
        ]
        baseline_events, baseline_as_of = _baseline_context(
            events=events,
            window_start=window_start,
            computed_at=computed_at,
        )
        baseline_profile = self.baseline_service.build_profile(
            baseline_events,
            as_of=baseline_as_of,
        )
        result = self.strategy.project(
            SpaceWebProjectionContext(
                recent_events=recent_events,
                baseline_profile=baseline_profile,
                computed_at=computed_at,
                category_label_overrides=self.category_label_overrides,
            )
        )
        return WellbeingSpaceWebPayload(
            computed_at=computed_at,
            range=requested_range,
            strategy_name=self.strategy.name,
            strategy_version=self.strategy.version,
            nodes=result.nodes,
            edges=result.edges,
            low_data=not baseline_profile.warmup_complete,
        )


def _baseline_context(
    *,
    events: list[AnalysisEvent],
    window_start: datetime,
    computed_at: datetime,
) -> tuple[list[AnalysisEvent], datetime]:
    before_window = [event for event in events if event.occurred_at < window_start]
    if before_window:
        return before_window, window_start
    return events, computed_at
