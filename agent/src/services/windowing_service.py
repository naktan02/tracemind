"""윈도우 요약 서비스.

ScoredEvent 배치를 프라이버시 안전 WindowSummary로 집계한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from shared.src.domain.entities.scored_event import ScoredEvent
from shared.src.domain.entities.window_summary import CategoryStats, WindowSummary


@dataclass(slots=True)
class WindowSummaryBuilder:
    """로컬 scored event 배치를 privacy-safe summary로 변환한다."""

    schema_version: str = "window_summary.v1"

    def build(
        self,
        scored_events: Sequence[ScoredEvent],
        *,
        age_band: str,
        summary_id: str | None = None,
    ) -> WindowSummary:
        if not age_band.strip():
            raise ValueError("age_band must not be empty.")
        if not scored_events:
            raise ValueError("scored_events must not be empty.")

        batch_started_at = min(event.occurred_at for event in scored_events)
        batch_ended_at = max(event.occurred_at for event in scored_events)

        totals: dict[str, float] = {}
        maxima: dict[str, float] = {}
        counts: dict[str, int] = {}

        for event in scored_events:
            for category, score in event.category_scores.items():
                totals[category] = totals.get(category, 0.0) + score
                maxima[category] = max(maxima.get(category, score), score)
                counts[category] = counts.get(category, 0) + 1

        if not counts:
            raise ValueError("At least one category score is required to build a summary.")

        category_stats = {
            category: CategoryStats(
                mean=totals[category] / counts[category],
                max=maxima[category],
                count=counts[category],
            )
            for category in sorted(counts)
        }

        return WindowSummary(
            schema_version=self.schema_version,
            summary_id=summary_id or str(uuid4()),
            age_band=age_band,
            batch_started_at=batch_started_at,
            batch_ended_at=batch_ended_at,
            event_count=len(scored_events),
            category_stats=category_stats,
        )


@dataclass(slots=True)
class WindowingService:
    """프라이버시를 보존하는 micro-batch 또는 rolling window 요약을 만든다."""

    window_days: int = 7
    _builder: WindowSummaryBuilder | None = None

    @property
    def builder(self) -> WindowSummaryBuilder:
        if self._builder is None:
            self._builder = WindowSummaryBuilder()
        return self._builder
