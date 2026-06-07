"""로컬 시계열 누적 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.inference.state import (
    BaselineProfile,
    PersonalizationState,
    TimeSeriesState,
)


@dataclass(slots=True)
class TimeSeriesConfig:
    """시계열 누적 계산 설정."""

    ewma_alpha: float = 0.4
    default_delta_threshold: float = 0.2


@dataclass(slots=True)
class TimeSeriesAccumulator:
    """현재 score를 기준선과 비교해 누적 상태를 갱신한다."""

    config: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)

    def update(
        self,
        *,
        analysis_event: AnalysisEvent,
        baseline_profile: BaselineProfile,
        personalization_state: PersonalizationState,
        previous_state: TimeSeriesState | None = None,
        state_version: str = "time_series_state.v1",
    ) -> TimeSeriesState:
        categories = set(analysis_event.category_scores)
        if previous_state is not None:
            categories.update(previous_state.latest_scores)
        categories.update(baseline_profile.category_means)
        categories.update(personalization_state.threshold_by_category)

        latest_scores: dict[str, float] = {}
        latest_deltas: dict[str, float] = {}
        ewma_scores: dict[str, float] = {}
        ewma_deltas: dict[str, float] = {}
        elevated_streaks: dict[str, int] = {}
        event_counts: dict[str, int] = {}

        alpha = self.config.ewma_alpha
        for category in sorted(categories):
            current_score = analysis_event.category_scores.get(
                category,
                previous_state.latest_scores.get(category, 0.0)
                if previous_state
                else 0.0,
            )
            baseline = baseline_profile.category_means.get(category, 0.0)
            delta = current_score - baseline

            previous_score = (
                previous_state.ewma_scores.get(category, current_score)
                if previous_state
                else current_score
            )
            previous_delta = (
                previous_state.ewma_deltas.get(category, delta)
                if previous_state
                else delta
            )
            threshold = personalization_state.threshold_by_category.get(
                category,
                self.config.default_delta_threshold,
            )
            previous_streak = (
                previous_state.elevated_streaks.get(category, 0)
                if previous_state
                else 0
            )
            previous_count = (
                previous_state.event_counts.get(category, 0) if previous_state else 0
            )

            latest_scores[category] = current_score
            latest_deltas[category] = delta
            ewma_scores[category] = (
                alpha * current_score + (1.0 - alpha) * previous_score
            )
            ewma_deltas[category] = alpha * delta + (1.0 - alpha) * previous_delta
            elevated_streaks[category] = (
                previous_streak + 1 if delta >= threshold else 0
            )
            event_counts[category] = previous_count + (
                1 if category in analysis_event.category_scores else 0
            )

        return TimeSeriesState(
            state_version=state_version,
            last_updated_at=analysis_event.occurred_at,
            latest_scores=latest_scores,
            latest_deltas=latest_deltas,
            ewma_scores=ewma_scores,
            ewma_deltas=ewma_deltas,
            elevated_streaks=elevated_streaks,
            event_counts=event_counts,
        )
