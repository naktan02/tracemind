"""부모용 상세 화면 접근을 위한 mock PIN service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agent.src.infrastructure.repositories.parent_auth_repository import (
    ParentAuthRepository,
    ParentAuthState,
)
from agent.src.infrastructure.repositories.wellbeing_settings_repository import (
    WellbeingSettingsRecord,
    WellbeingSettingsRepository,
)
from shared.src.contracts.wellbeing_signal_contracts import (
    ParentUnlockResponsePayload,
)


@dataclass(slots=True)
class ParentAuthService:
    """부모용 PIN 잠금 흐름을 제공한다.

    현재 단계에서는 repository와 설정 저장소를 우선 source of truth로 사용하고,
    주입되지 않았을 때만 in-memory fallback으로 동작한다.
    """

    pin_code: str = "1234"
    repository: ParentAuthRepository | None = None
    settings_repository: WellbeingSettingsRepository | None = None
    max_attempts: int = 5
    lock_minutes: int = 10
    session_minutes: int = 15
    _failed_attempts: int = field(default=0, init=False)
    _locked_until: datetime | None = field(default=None, init=False)

    def unlock(self, *, pin: str) -> ParentUnlockResponsePayload:
        now = datetime.now(tz=timezone.utc)
        settings = self._load_settings()
        auth_state = self._load_auth_state(now=now)
        effective_locked_until = (
            auth_state.locked_until if auth_state is not None else self._locked_until
        )

        if effective_locked_until is not None and now < effective_locked_until:
            return ParentUnlockResponsePayload(
                granted=False,
                remaining_attempts=0,
                locked_until=effective_locked_until,
            )

        if self._verify_pin(pin=pin, auth_state=auth_state):
            self._failed_attempts = 0
            self._locked_until = None
            self._persist_auth_state(
                ParentAuthState(
                    pin_hash=self._configured_pin_hash(),
                    failed_attempt_count=0,
                    locked_until=None,
                    updated_at=now,
                )
            )
            return ParentUnlockResponsePayload(
                granted=True,
                session_token=f"parent-session-{uuid4().hex[:12]}",
                session_expires_at=now
                + timedelta(minutes=settings.parent_session_minutes),
                remaining_attempts=settings.parent_max_attempts,
            )

        current_failed_attempts = (
            auth_state.failed_attempt_count
            if auth_state is not None
            else self._failed_attempts
        )
        current_failed_attempts += 1
        self._failed_attempts = current_failed_attempts
        remaining_attempts = max(
            settings.parent_max_attempts - current_failed_attempts,
            0,
        )
        if remaining_attempts == 0:
            self._locked_until = now + timedelta(minutes=settings.parent_lock_minutes)

        persisted_state = ParentAuthState(
            pin_hash=self._configured_pin_hash(),
            failed_attempt_count=current_failed_attempts,
            locked_until=self._locked_until,
            updated_at=now,
        )
        self._persist_auth_state(persisted_state)

        return ParentUnlockResponsePayload(
            granted=False,
            remaining_attempts=remaining_attempts,
            locked_until=self._locked_until,
        )

    def _configured_pin_hash(self) -> str:
        return hashlib.sha256(self.pin_code.encode("utf-8")).hexdigest()

    def _load_settings(self):
        if self.settings_repository is None:
            return WellbeingSettingsRecord(
                parent_session_minutes=self.session_minutes,
                parent_lock_minutes=self.lock_minutes,
                parent_max_attempts=self.max_attempts,
            )
        return self.settings_repository.load_or_default()

    def _load_auth_state(self, *, now: datetime) -> ParentAuthState | None:
        if self.repository is None:
            return None
        auth_state = self.repository.load_state()
        if auth_state is None:
            auth_state = ParentAuthState(
                pin_hash=self._configured_pin_hash(),
                failed_attempt_count=0,
                locked_until=None,
                updated_at=now,
            )
            self.repository.save_state(auth_state)
        return auth_state

    def _verify_pin(
        self,
        *,
        pin: str,
        auth_state: ParentAuthState | None,
    ) -> bool:
        candidate_hash = hashlib.sha256(pin.encode("utf-8")).hexdigest()
        if auth_state is None:
            return candidate_hash == self._configured_pin_hash()
        return candidate_hash == auth_state.pin_hash

    def _persist_auth_state(self, state: ParentAuthState) -> None:
        if self.repository is not None:
            self.repository.save_state(state)
