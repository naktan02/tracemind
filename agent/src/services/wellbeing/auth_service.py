"""부모용 상세 화면 접근을 위한 mock PIN service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from shared.src.contracts.wellbeing_signal_contracts import (
    ParentUnlockResponsePayload,
)


@dataclass(slots=True)
class ParentAuthService:
    """부모용 PIN 잠금 흐름을 제공한다.

    MVP 1차 구현은 in-memory 상태만 사용한다.
    다음 단계에서는 repository 기반으로 교체한다.
    """

    pin_code: str = "1234"
    max_attempts: int = 5
    lock_minutes: int = 10
    session_minutes: int = 15
    _failed_attempts: int = field(default=0, init=False)
    _locked_until: datetime | None = field(default=None, init=False)

    def unlock(self, *, pin: str) -> ParentUnlockResponsePayload:
        now = datetime.now(tz=timezone.utc)
        if self._locked_until is not None and now < self._locked_until:
            return ParentUnlockResponsePayload(
                granted=False,
                remaining_attempts=0,
                locked_until=self._locked_until,
            )

        if pin == self.pin_code:
            self._failed_attempts = 0
            self._locked_until = None
            return ParentUnlockResponsePayload(
                granted=True,
                session_token=f"parent-session-{uuid4().hex[:12]}",
                session_expires_at=now + timedelta(minutes=self.session_minutes),
                remaining_attempts=self.max_attempts,
            )

        self._failed_attempts += 1
        remaining_attempts = max(self.max_attempts - self._failed_attempts, 0)
        if remaining_attempts == 0:
            self._locked_until = now + timedelta(minutes=self.lock_minutes)

        return ParentUnlockResponsePayload(
            granted=False,
            remaining_attempts=remaining_attempts,
            locked_until=self._locked_until,
        )
