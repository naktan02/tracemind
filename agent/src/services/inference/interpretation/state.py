"""로컬 inference 상태 entity 모음."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class BaselineProfile:
    """한 로컬 사용자에 대한 rolling self-baseline feature."""

    profile_version: str
    warmup_complete: bool
    observed_days: int = 0
    event_count: int = 0
    category_means: dict[str, float] = field(default_factory=dict)
    category_sigmas: dict[str, float] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    category_latest: dict[str, float] = field(default_factory=dict)
    persistence_days: int = 0
    computed_at: datetime | None = None


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
