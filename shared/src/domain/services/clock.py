"""시간 제공자 추상화."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """현재 시각을 제공하는 인터페이스."""

    def now(self) -> datetime:
        """현재 시각을 반환한다."""


@dataclass(slots=True)
class SystemUtcClock:
    """UTC 기준 현재 시각을 제공하는 기본 clock."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(slots=True)
class FixedClock:
    """항상 고정된 시각을 반환하는 테스트용 clock."""

    current_time: datetime

    def now(self) -> datetime:
        return self.current_time
