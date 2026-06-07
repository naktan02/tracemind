"""Family access contracts tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent.src.contracts.family_access_contracts import (
    FamilyAccessMode,
    FamilyAccessRole,
    FamilySetupRequestPayload,
    FamilySetupStatusPayload,
    FamilyUnlockRequestPayload,
    FamilyUnlockResponsePayload,
)


def test_family_setup_status_defaults_to_local_mode() -> None:
    payload = FamilySetupStatusPayload(
        is_setup_complete=True,
        configured_roles=(FamilyAccessRole.CHILD, FamilyAccessRole.PARENT),
    )

    assert payload.access_mode == FamilyAccessMode.THIS_DEVICE_ONLY


def test_family_setup_request_requires_numeric_pins() -> None:
    with pytest.raises(ValidationError):
        FamilySetupRequestPayload(
            child_pin="12ab",
            parent_pin="1234",
        )


def test_family_unlock_response_round_trips_role_and_session() -> None:
    payload = FamilyUnlockResponsePayload(
        role=FamilyAccessRole.PARENT,
        granted=True,
        session_token="session-1",
        session_expires_at=datetime(2026, 4, 24, 10, 30, tzinfo=timezone.utc),
        remaining_attempts=5,
    )

    assert payload.role == FamilyAccessRole.PARENT
    assert payload.granted is True


def test_family_unlock_request_requires_role() -> None:
    request = FamilyUnlockRequestPayload(
        role=FamilyAccessRole.CHILD,
        pin="1234",
    )

    assert request.role == FamilyAccessRole.CHILD
