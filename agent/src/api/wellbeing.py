"""가족용 확장 프로그램이 읽는 wellbeing API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from agent.src.services.wellbeing.auth_service import ParentAuthService
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from agent.src.services.wellbeing.timeseries_service import WellbeingTimeseriesService
from shared.src.contracts.wellbeing_signal_contracts import (
    ParentUnlockRequestPayload,
    ParentUnlockResponsePayload,
    WellbeingSignalRange,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTimeseriesPayload,
)

router = APIRouter(prefix="/api/v1", tags=["wellbeing"])


class SystemHealthResponse(BaseModel):
    """확장 프로그램이 로컬 프로그램 상태를 확인할 때 쓰는 응답."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    wellbeing_api_ready: bool


def get_wellbeing_summary_service(request: Request) -> WellbeingSummaryService:
    service = getattr(request.app.state, "wellbeing_summary_service", None)
    if service is None:
        raise RuntimeError(
            "WellbeingSummaryService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.wellbeing_summary_service를 설정하세요."
        )
    return service


def get_wellbeing_timeseries_service(
    request: Request,
) -> WellbeingTimeseriesService:
    service = getattr(request.app.state, "wellbeing_timeseries_service", None)
    if service is None:
        raise RuntimeError(
            "WellbeingTimeseriesService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.wellbeing_timeseries_service를 설정하세요."
        )
    return service


def get_parent_auth_service(request: Request) -> ParentAuthService:
    service = getattr(request.app.state, "parent_auth_service", None)
    if service is None:
        raise RuntimeError(
            "ParentAuthService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.parent_auth_service를 설정하세요."
        )
    return service


SummaryServiceDep = Annotated[
    WellbeingSummaryService,
    Depends(get_wellbeing_summary_service),
]
TimeseriesServiceDep = Annotated[
    WellbeingTimeseriesService,
    Depends(get_wellbeing_timeseries_service),
]
ParentAuthServiceDep = Annotated[
    ParentAuthService,
    Depends(get_parent_auth_service),
]


@router.get(
    "/wellbeing/summary",
    response_model=WellbeingSignalSummaryPayload,
)
def get_wellbeing_summary(
    summary_service: SummaryServiceDep,
) -> WellbeingSignalSummaryPayload:
    """현재 wellbeing signal 한 건을 반환한다."""
    return summary_service.get_current_summary()


@router.get(
    "/wellbeing/timeseries",
    response_model=WellbeingSignalTimeseriesPayload,
)
def get_wellbeing_timeseries(
    range: WellbeingSignalRange,
    timeseries_service: TimeseriesServiceDep,
) -> WellbeingSignalTimeseriesPayload:
    """부모용 상세 화면의 전체 wellbeing signal 추이를 반환한다."""
    return timeseries_service.get_timeseries(requested_range=range)


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
