"""관리자용 strategy 전환 API 라우터."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from main_server.src.services.federation.strategy.active_strategy_service import (
    ActiveStrategyService,
    StrategyValidationError,
)
from main_server.src.services.federation.strategy.models import ActiveStrategyConfig
from methods.federated_ssl.registry import list_federated_ssl_method_descriptors

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ------------------------------------------------------------------ #
# 요청/응답 모델                                                         #
# ------------------------------------------------------------------ #


class SwitchStrategyRequest(BaseModel):
    """strategy 전환 요청."""

    model_config = ConfigDict(extra="forbid")

    ssl_method: str | None = Field(
        default=None,
        description="로컬 SSL objective 이름. 없으면 현재 값 유지.",
        examples=["fixmatch_usb_v1", "flexmatch_usb_v1"],
    )
    aggregation_backend: str | None = Field(
        default=None,
        description="서버 집계 backend 이름. 없으면 현재 값 유지.",
        examples=["fedavg"],
    )
    notes: str | None = Field(
        default=None,
        description="전환 이유 메모.",
    )


class StrategyResponse(BaseModel):
    """active strategy 응답."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    ssl_method: str
    aggregation_backend: str
    activated_at: str
    notes: str | None


class MethodInfo(BaseModel):
    """등록된 method 정보."""

    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str | None
    implementation_status: str
    live_server_supported: bool
    live_agent_supported: bool
    simulation_supported: bool
    requires_custom_client_runtime: bool


# ------------------------------------------------------------------ #
# 의존성                                                                #
# ------------------------------------------------------------------ #


def get_active_strategy_service(request: Request) -> ActiveStrategyService:
    """app.state에서 ActiveStrategyService를 읽는다."""
    service = getattr(request.app.state, "active_strategy_service", None)
    if service is None:
        raise RuntimeError("ActiveStrategyService가 app.state에 설정되지 않았습니다.")
    return service


ActiveStrategyServiceDep = Annotated[
    ActiveStrategyService, Depends(get_active_strategy_service)
]


# ------------------------------------------------------------------ #
# 유틸                                                                  #
# ------------------------------------------------------------------ #


def _config_to_response(config: ActiveStrategyConfig) -> StrategyResponse:
    return StrategyResponse(
        schema_version=config.schema_version,
        ssl_method=config.ssl_method,
        aggregation_backend=config.aggregation_backend,
        activated_at=config.activated_at.isoformat(),
        notes=config.notes,
    )


# ------------------------------------------------------------------ #
# 엔드포인트                                                             #
# ------------------------------------------------------------------ #


@router.post(
    "/strategy",
    response_model=StrategyResponse,
    status_code=status.HTTP_200_OK,
    summary="FL strategy 전환",
    description=(
        "운영 중 FL 방법론을 전환한다. "
        "다음 라운드 open 시부터 새 strategy가 자동으로 적용된다."
    ),
)
def switch_strategy(
    body: SwitchStrategyRequest,
    service: ActiveStrategyServiceDep,
) -> StrategyResponse:
    try:
        config = service.switch(
            ssl_method=body.ssl_method,
            aggregation_backend=body.aggregation_backend,
            notes=body.notes,
        )
        return _config_to_response(config)
    except StrategyValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get(
    "/strategy/current",
    response_model=StrategyResponse,
    status_code=status.HTTP_200_OK,
    summary="현재 active strategy 조회",
)
def get_current_strategy(service: ActiveStrategyServiceDep) -> StrategyResponse:
    return _config_to_response(service.get_active_strategy())


@router.get(
    "/strategy/history",
    response_model=list[StrategyResponse],
    status_code=status.HTTP_200_OK,
    summary="strategy 전환 이력 조회",
)
def get_strategy_history(service: ActiveStrategyServiceDep) -> list[StrategyResponse]:
    return [_config_to_response(c) for c in service.get_history()]


@router.get(
    "/methods",
    response_model=list[MethodInfo],
    status_code=status.HTTP_200_OK,
    summary="등록된 FL method 목록",
    description="registry에 등록된 모든 FL SSL method descriptor를 반환한다.",
)
def list_methods() -> list[MethodInfo]:
    descriptors = list_federated_ssl_method_descriptors()
    return [
        MethodInfo(
            name=d.name,
            display_name=d.display_name,
            implementation_status=d.implementation_status,
            live_server_supported=d.runtime_capabilities.live_server_supported,
            live_agent_supported=d.runtime_capabilities.live_agent_supported,
            simulation_supported=d.runtime_capabilities.simulation_supported,
            requires_custom_client_runtime=d.runtime_capabilities.requires_custom_client_runtime,
        )
        for d in descriptors
    ]
