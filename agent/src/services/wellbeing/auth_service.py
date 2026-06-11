"""구형 parent unlock endpoint를 family_access 경계 위에 유지하는 adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from agent.src.contracts.family_access_contracts import FamilyAccessRole
from agent.src.contracts.wellbeing_signal_contracts import (
    ParentUnlockResponsePayload,
)
from agent.src.services.wellbeing.family_access_service import FamilyAccessService


@dataclass(slots=True)
class ParentAuthService:
    """기존 parent unlock API를 family_access parent role로 위임한다."""

    family_access_service: FamilyAccessService
    _last_parent_session_expires_at: datetime | None = field(default=None, init=False)

    def unlock(self, *, pin: str) -> ParentUnlockResponsePayload:
        response = self.family_access_service.unlock(
            role=FamilyAccessRole.PARENT,
            pin=pin,
        )
        self._last_parent_session_expires_at = response.session_expires_at
        return ParentUnlockResponsePayload(
            granted=response.granted,
            session_token=response.session_token,
            session_expires_at=response.session_expires_at,
            remaining_attempts=response.remaining_attempts,
            locked_until=response.locked_until,
        )
