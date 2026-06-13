"""가족용 확장 프로그램의 setup/auth 경계를 관리한다."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agent.src.contracts.family_access_contracts import (
    FamilyAccessRole,
    FamilySetupResponsePayload,
    FamilySetupStatusPayload,
    FamilyUnlockResponsePayload,
)
from agent.src.infrastructure.repositories.family_access_repository import (
    FamilyAccessRepository,
    FamilyAccessState,
)
from agent.src.infrastructure.repositories.wellbeing_settings_repository import (
    WellbeingSettingsRecord,
    WellbeingSettingsRepository,
)


class FamilyAccessSetupAlreadyCompletedError(RuntimeError):
    """이미 최초 setup이 끝난 상태에서 다시 setup을 요청했다."""


@dataclass(frozen=True, slots=True)
class _RoleAccessPolicy:
    session_minutes: int
    lock_minutes: int
    max_attempts: int


@dataclass(slots=True)
class FamilyAccessService:
    """child/parent 공통 setup 및 PIN 잠금 흐름을 제공한다."""

    repository: FamilyAccessRepository
    settings_repository: WellbeingSettingsRepository | None = None

    def get_setup_status(self) -> FamilySetupStatusPayload:
        configured_roles = self.repository.list_configured_roles()
        return FamilySetupStatusPayload(
            is_setup_complete=_is_setup_complete(configured_roles),
            configured_roles=configured_roles,
        )

    def create_initial_setup(
        self,
        *,
        child_pin: str,
        parent_pin: str,
    ) -> FamilySetupResponsePayload:
        status = self.get_setup_status()
        if status.is_setup_complete:
            raise FamilyAccessSetupAlreadyCompletedError(
                "초기 family access setup이 이미 완료된 상태입니다."
            )

        now = datetime.now(tz=timezone.utc)
        for role, pin in (
            (FamilyAccessRole.CHILD, child_pin),
            (FamilyAccessRole.PARENT, parent_pin),
        ):
            self.repository.save_state(
                FamilyAccessState(
                    role=role,
                    pin_hash=_hash_pin(pin),
                    failed_attempt_count=0,
                    locked_until=None,
                    updated_at=now,
                )
            )

        configured_roles = self.repository.list_configured_roles()
        return FamilySetupResponsePayload(
            is_setup_complete=_is_setup_complete(configured_roles),
            configured_roles=configured_roles,
        )

    def unlock(
        self,
        *,
        role: FamilyAccessRole,
        pin: str,
    ) -> FamilyUnlockResponsePayload:
        now = datetime.now(tz=timezone.utc)
        settings = self._load_settings()
        policy = _policy_for_role(settings=settings, role=role)
        state = self.repository.load_state(role)
        if state is None:
            return FamilyUnlockResponsePayload(
                role=role,
                granted=False,
                remaining_attempts=0,
            )

        if state.locked_until is not None and now < state.locked_until:
            return FamilyUnlockResponsePayload(
                role=role,
                granted=False,
                remaining_attempts=0,
                locked_until=state.locked_until,
            )

        if _hash_pin(pin) == state.pin_hash:
            self.repository.save_state(
                FamilyAccessState(
                    role=role,
                    pin_hash=state.pin_hash,
                    failed_attempt_count=0,
                    locked_until=None,
                    updated_at=now,
                )
            )
            return FamilyUnlockResponsePayload(
                role=role,
                granted=True,
                session_token=f"{role.value}-session-{uuid4().hex[:12]}",
                session_expires_at=now + timedelta(minutes=policy.session_minutes),
                remaining_attempts=policy.max_attempts,
            )

        failed_attempt_count = state.failed_attempt_count + 1
        remaining_attempts = max(policy.max_attempts - failed_attempt_count, 0)
        locked_until = (
            now + timedelta(minutes=policy.lock_minutes)
            if remaining_attempts == 0
            else None
        )
        self.repository.save_state(
            FamilyAccessState(
                role=role,
                pin_hash=state.pin_hash,
                failed_attempt_count=failed_attempt_count,
                locked_until=locked_until,
                updated_at=now,
            )
        )
        return FamilyUnlockResponsePayload(
            role=role,
            granted=False,
            remaining_attempts=remaining_attempts,
            locked_until=locked_until,
        )

    def _load_settings(self) -> WellbeingSettingsRecord:
        if self.settings_repository is None:
            return WellbeingSettingsRecord()
        return self.settings_repository.load_or_default()


def _hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def _is_setup_complete(configured_roles: tuple[FamilyAccessRole, ...]) -> bool:
    return (
        FamilyAccessRole.CHILD in configured_roles
        and FamilyAccessRole.PARENT in configured_roles
    )


def _policy_for_role(
    *,
    settings: WellbeingSettingsRecord,
    role: FamilyAccessRole,
) -> _RoleAccessPolicy:
    if role is FamilyAccessRole.CHILD:
        return _RoleAccessPolicy(
            session_minutes=settings.child_session_minutes,
            lock_minutes=settings.child_lock_minutes,
            max_attempts=settings.child_max_attempts,
        )
    return _RoleAccessPolicy(
        session_minutes=settings.parent_session_minutes,
        lock_minutes=settings.parent_lock_minutes,
        max_attempts=settings.parent_max_attempts,
    )
