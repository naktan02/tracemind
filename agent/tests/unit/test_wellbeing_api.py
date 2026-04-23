"""Wellbeing API unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.middleware.cors import CORSMiddleware

from agent.src.api import family_access as family_access_api
from agent.src.api import wellbeing as wellbeing_api
from agent.src.api.main import (
    DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS,
    app,
    create_app,
    load_family_extension_allowed_origins_from_env,
)
from agent.src.infrastructure.repositories.family_access_repository import (
    FamilyAccessRepository,
)
from agent.src.services.wellbeing.family_access_service import FamilyAccessService
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from agent.src.services.wellbeing.timeseries_service import (
    WellbeingTimeseriesService,
)
from shared.src.contracts.family_access_contracts import (
    FamilyAccessRole,
    FamilySetupRequestPayload,
    FamilyUnlockRequestPayload,
)
from shared.src.contracts.wellbeing_signal_contracts import (
    ParentUnlockRequestPayload,
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalRange,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)


def test_wellbeing_summary_api_returns_service_payload() -> None:
    payload = WellbeingSignalSummaryPayload(
        computed_at=datetime(2026, 4, 24, 10, 30, tzinfo=timezone.utc),
        signal_score=58.0,
        signal_level=WellbeingSignalLevel.MODERATE,
        signal_label="관찰 필요",
        trend=WellbeingSignalTrend.STEADY,
        summary="최근 상태가 비교적 안정적으로 유지되고 있습니다.",
        action_tip="오늘 저녁에 짧게 안부를 물어보세요.",
        confidence=WellbeingSignalConfidence.MEDIUM,
        low_data=False,
    )

    response = wellbeing_api.get_wellbeing_summary(
        summary_service=WellbeingSummaryService(_mock_payload=payload)
    )

    assert response == payload


def test_wellbeing_timeseries_api_returns_requested_range() -> None:
    response = wellbeing_api.get_wellbeing_timeseries(
        range=WellbeingSignalRange.LAST_14_DAYS,
        timeseries_service=WellbeingTimeseriesService(),
    )

    assert response.range == WellbeingSignalRange.LAST_14_DAYS
    assert len(response.points) == 14


def test_parent_unlock_api_grants_valid_pin() -> None:
    family_access_service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=_temporary_test_db("parent-grant"))
    )
    family_access_service.create_initial_setup(child_pin="1111", parent_pin="1234")
    response = wellbeing_api.unlock_parent_view(
        ParentUnlockRequestPayload(pin="1234"),
        auth_service=wellbeing_api.ParentAuthService(
            family_access_service=family_access_service
        ),
    )

    assert response.granted is True
    assert response.session_token is not None


def test_parent_unlock_api_rejects_invalid_pin() -> None:
    family_access_service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=_temporary_test_db("parent-reject"))
    )
    family_access_service.create_initial_setup(child_pin="1111", parent_pin="1234")
    response = wellbeing_api.unlock_parent_view(
        ParentUnlockRequestPayload(pin="9999"),
        auth_service=wellbeing_api.ParentAuthService(
            family_access_service=family_access_service
        ),
    )

    assert response.granted is False
    assert response.remaining_attempts == 4


def test_family_setup_status_api_reflects_setup_completion() -> None:
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=_temporary_test_db("setup-status"))
    )

    before = family_access_api.get_family_setup_status(service)
    service.create_initial_setup(child_pin="1111", parent_pin="2222")
    after = family_access_api.get_family_setup_status(service)

    assert before.is_setup_complete is False
    assert after.is_setup_complete is True


def test_family_setup_api_creates_local_only_setup() -> None:
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=_temporary_test_db("setup-create"))
    )

    response = family_access_api.create_family_setup(
        FamilySetupRequestPayload(child_pin="1111", parent_pin="2222"),
        service,
    )

    assert response.is_setup_complete is True
    assert response.configured_roles == (
        FamilyAccessRole.CHILD,
        FamilyAccessRole.PARENT,
    )


def test_family_unlock_api_uses_requested_role() -> None:
    service = FamilyAccessService(
        repository=FamilyAccessRepository(db_path=_temporary_test_db("family-unlock"))
    )
    service.create_initial_setup(child_pin="1111", parent_pin="2222")

    response = family_access_api.unlock_family_role(
        FamilyUnlockRequestPayload(role=FamilyAccessRole.CHILD, pin="1111"),
        service,
    )

    assert response.granted is True
    assert response.role == FamilyAccessRole.CHILD


def test_wellbeing_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/family/setup/status" in route_paths
    assert "/api/v1/family/setup" in route_paths
    assert "/api/v1/family/unlock" in route_paths
    assert "/api/v1/wellbeing/summary" in route_paths
    assert "/api/v1/wellbeing/timeseries" in route_paths
    assert "/api/v1/parent/unlock" in route_paths
    assert "/api/v1/system/health" in route_paths


def test_agent_app_uses_family_extension_default_origins() -> None:
    local_app = create_app()

    cors_middleware = next(
        middleware
        for middleware in local_app.user_middleware
        if middleware.cls is CORSMiddleware
    )

    assert cors_middleware.kwargs["allow_origins"] == list(
        DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS
    )


def test_family_extension_origin_loader_reads_environment_mapping() -> None:
    origins = load_family_extension_allowed_origins_from_env(
        environ={
            "FAMILY_EXTENSION_ALLOWED_ORIGINS": (
                "http://localhost:5174, https://family.example.com "
            )
        }
    )

    assert origins == (
        "http://localhost:5174",
        "https://family.example.com",
    )


def _temporary_test_db(name: str) -> Path:
    return Path(f"/tmp/{name}-{uuid4().hex}.wellbeing.db")
