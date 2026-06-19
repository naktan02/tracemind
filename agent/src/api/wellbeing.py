"""가족용 확장 프로그램이 읽는 wellbeing API."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from agent.src.api.dependencies import (
    ParentAuthServiceDep,
    WellbeingSpaceWebServiceDep,
    WellbeingSummaryServiceDep,
    WellbeingTimeseriesServiceDep,
)
from agent.src.contracts.wellbeing_signal_contracts import (
    ParentUnlockRequestPayload,
    ParentUnlockResponsePayload,
    WellbeingSignalRange,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTimeseriesPayload,
)
from agent.src.contracts.wellbeing_space_web_contracts import (
    WellbeingSpaceWebPayload,
)
from agent.src.features.wellbeing.family_access.parent_auth_adapter import (  # noqa: F401
    ParentAuthService,
)

router = APIRouter(prefix="/api/v1", tags=["wellbeing"])


class SystemHealthResponse(BaseModel):
    """확장 프로그램이 로컬 프로그램 상태를 확인할 때 쓰는 응답."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    wellbeing_api_ready: bool


@router.get(
    "/wellbeing/summary",
    response_model=WellbeingSignalSummaryPayload,
)
def get_wellbeing_summary(
    summary_service: WellbeingSummaryServiceDep,
) -> WellbeingSignalSummaryPayload:
    """현재 wellbeing signal 한 건을 반환한다."""
    return summary_service.get_current_summary()


@router.get(
    "/wellbeing/timeseries",
    response_model=WellbeingSignalTimeseriesPayload,
)
def get_wellbeing_timeseries(
    range: WellbeingSignalRange,
    timeseries_service: WellbeingTimeseriesServiceDep,
) -> WellbeingSignalTimeseriesPayload:
    """부모용 상세 화면의 전체 wellbeing signal 추이를 반환한다."""
    return timeseries_service.get_timeseries(requested_range=range)


@router.get(
    "/wellbeing/space-web",
    response_model=WellbeingSpaceWebPayload,
)
def get_wellbeing_space_web(
    range: WellbeingSignalRange,
    space_web_service: WellbeingSpaceWebServiceDep,
) -> WellbeingSpaceWebPayload:
    """아이용 분석 화면의 wellbeing space-web graph를 반환한다."""
    return space_web_service.get_space_web(requested_range=range)


@router.post(
    "/parent/unlock",
    response_model=ParentUnlockResponsePayload,
)
def unlock_parent_view(
    request: ParentUnlockRequestPayload,
    auth_service: ParentAuthServiceDep,
) -> ParentUnlockResponsePayload:
    """부모용 상세 화면 접근을 위한 PIN 검증."""
    return auth_service.unlock(pin=request.pin)


@router.get(
    "/system/health",
    response_model=SystemHealthResponse,
)
def get_system_health() -> SystemHealthResponse:
    """확장 프로그램이 로컬 프로그램 연결 상태를 확인한다."""
    return SystemHealthResponse(
        status="ok",
        service="agent",
        wellbeing_api_ready=True,
    )
