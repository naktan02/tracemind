"""개인 기준선 계산 서비스."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from shared.src.domain.entities.baseline_profile import BaselineProfile
from shared.src.domain.entities.scored_event import ScoredEvent


@dataclass(slots=True)
class BaselineConfig:
    """기준선 계산 설정."""

    lookback_days: int = 28
    warmup_days: int = 14
    warmup_events: int = 5
    minimum_sigma: float = 0.05


@dataclass(slots=True)
class BaselineService:
    """과거 scored event로부터 개인 baseline profile을 계산한다."""

    config: BaselineConfig = field(default_factory=BaselineConfig)

    def build_profile(
        self,
        scored_events: Sequence[ScoredEvent],
        *,
        as_of: datetime | None = None,
        profile_version: str = "baseline_profile.v1",
    ) -> BaselineProfile:
        if not scored_events:
            return BaselineProfile(
                profile_version=profile_version,
                warmup_complete=False,
                computed_at=as_of,
            )

        effective_as_of = as_of or max(event.occurred_at for event in scored_events)
        filtered_events = self._filter_events(scored_events, as_of=effective_as_of)
        if not filtered_events:
            return BaselineProfile(
                profile_version=profile_version,
                warmup_complete=False,
                computed_at=effective_as_of,
            )

        totals: dict[str, float] = defaultdict(float)
        squared_totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        latest: dict[str, tuple[datetime, float]] = {}

        for event in filtered_events:
            for category, score in event.category_scores.items():
                totals[category] += score
                squared_totals[category] += score * score
                counts[category] += 1
                previous = latest.get(category)
                if previous is None or previous[0] <= event.occurred_at:
                    latest[category] = (event.occurred_at, score)

        category_means: dict[str, float] = {}
        category_sigmas: dict[str, float] = {}
        category_latest: dict[str, float] = {}

        for category in sorted(counts):
            count = counts[category]
            mean = totals[category] / count
            variance = max((squared_totals[category] / count) - (mean * mean), 0.0)
            sigma = max(math.sqrt(variance), self.config.minimum_sigma)
            category_means[category] = mean
            category_sigmas[category] = sigma
            category_latest[category] = latest[category][1]

        observed_dates = {event.occurred_at.date() for event in filtered_events}
        observed_days = len(observed_dates)
        warmup_complete = (
            len(filtered_events) >= self.config.warmup_events
            and observed_days >= self.config.warmup_days
        )

        return BaselineProfile(
            profile_version=profile_version,
            warmup_complete=warmup_complete,
            observed_days=observed_days,
            event_count=len(filtered_events),
            category_means=category_means,
            category_sigmas=category_sigmas,
            category_counts=dict(counts),
            category_latest=category_latest,
            persistence_days=observed_days,
            computed_at=effective_as_of,
        )

    def _filter_events(
        self,
        scored_events: Sequence[ScoredEvent],
        *,
        as_of: datetime,
    ) -> list[ScoredEvent]:
        lower_bound = as_of - timedelta(days=self.config.lookback_days)
        return [
            event
            for event in scored_events
            if lower_bound <= event.occurred_at <= as_of
        ]
