"""개인 기준선 프로필."""

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
