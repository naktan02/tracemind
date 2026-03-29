"""로컬 시계열 누적 상태."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class TimeSeriesState:
    """이벤트 점수 흐름을 누적해 persistence를 계산하는 로컬 상태."""

    state_version: str
    last_updated_at: datetime | None = None
    latest_scores: dict[str, float] = field(default_factory=dict)
    latest_deltas: dict[str, float] = field(default_factory=dict)
    ewma_scores: dict[str, float] = field(default_factory=dict)
    ewma_deltas: dict[str, float] = field(default_factory=dict)
    elevated_streaks: dict[str, int] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
