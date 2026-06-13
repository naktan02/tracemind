"""Family access service unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.src.contracts.family_access_contracts import FamilyAccessRole
from agent.src.infrastructure.repositories.family_access_repository import (
    FamilyAccessRepository,
)
from agent.src.infrastructure.repositories.wellbeing_settings_repository import (
    WellbeingSettingsRecord,
    WellbeingSettingsRepository,
)
from agent.src.services.wellbeing.family_access.service import (
    FamilyAccessService,
    FamilyAccessSetupAlreadyCompletedError,
)


def test_family_access_service_reports_setup_incomplete_before_initial_setup(
    tmp_path: Path,
) -> None:
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=tmp_path / "wellbeing.db")
    )

    status = service.get_setup_status()

    assert status.is_setup_complete is False
    assert status.configured_roles == ()


def test_family_access_service_creates_initial_setup_for_child_and_parent(
    tmp_path: Path,
) -> None:
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=tmp_path / "wellbeing.db")
    )

    response = service.create_initial_setup(child_pin="1111", parent_pin="2222")

    assert response.is_setup_complete is True
    assert response.configured_roles == (
        FamilyAccessRole.CHILD,
        FamilyAccessRole.PARENT,
    )


def test_family_access_service_rejects_repeated_initial_setup(tmp_path: Path) -> None:
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=tmp_path / "wellbeing.db")
    )
    service.create_initial_setup(child_pin="1111", parent_pin="2222")

    with pytest.raises(FamilyAccessSetupAlreadyCompletedError):
        service.create_initial_setup(child_pin="3333", parent_pin="4444")


def test_family_access_service_unlocks_child_with_role_specific_policy(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wellbeing.db"
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=db_path),
        settings_repository=WellbeingSettingsRepository(db_path=db_path),
    )
    service.settings_repository.save_settings(
        WellbeingSettingsRecord(
            child_session_minutes=7,
            child_lock_minutes=2,
            child_max_attempts=2,
            parent_session_minutes=15,
            parent_lock_minutes=10,
            parent_max_attempts=5,
        )
    )
    service.create_initial_setup(child_pin="1111", parent_pin="2222")

    response = service.unlock(role=FamilyAccessRole.CHILD, pin="1111")

    assert response.granted is True
    assert response.role == FamilyAccessRole.CHILD
    assert response.session_expires_at is not None


def test_family_access_service_locks_after_max_attempts_for_role(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wellbeing.db"
    settings_repository = WellbeingSettingsRepository(db_path=db_path)
    settings_repository.save_settings(
        WellbeingSettingsRecord(
            child_max_attempts=2,
            child_lock_minutes=1,
            parent_max_attempts=5,
        )
    )
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=db_path),
        settings_repository=settings_repository,
    )
    service.create_initial_setup(child_pin="1111", parent_pin="2222")

    first = service.unlock(role=FamilyAccessRole.CHILD, pin="9999")
    second = service.unlock(role=FamilyAccessRole.CHILD, pin="9999")

    assert first.remaining_attempts == 1
    assert second.remaining_attempts == 0
    assert second.locked_until is not None
