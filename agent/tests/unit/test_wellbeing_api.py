"""Wellbeing API unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.src.api import wellbeing as wellbeing_api
from agent.src.api.main import app
from agent.src.services.wellbeing.auth_service import ParentAuthService
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from agent.src.services.wellbeing.timeseries_service import (
    WellbeingTimeseriesService,
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
    response = wellbeing_api.unlock_parent_view(
        ParentUnlockRequestPayload(pin="1234"),
        auth_service=ParentAuthService(pin_code="1234"),
    )

    assert response.granted is True
    assert response.session_token is not None


def test_parent_unlock_api_rejects_invalid_pin() -> None:
    response = wellbeing_api.unlock_parent_view(
        ParentUnlockRequestPayload(pin="9999"),
        auth_service=ParentAuthService(pin_code="1234"),
    )

    assert response.granted is False
    assert response.remaining_attempts == 4


def test_wellbeing_router_is_registered_on_agent_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/wellbeing/summary" in route_paths
    assert "/api/v1/wellbeing/timeseries" in route_paths
    assert "/api/v1/parent/unlock" in route_paths
    assert "/api/v1/system/health" in route_paths
