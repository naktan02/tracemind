"""Agent runtime profile API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from main_server.src.services.agent_runtime_profile_service import (
    AgentRuntimeProfileService,
)
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
    AgentRuntimeProfileValidationRequestPayload,
    AgentRuntimeProfileValidationResponsePayload,
)

router = APIRouter(prefix="/api/v1/agent-runtime-profile", tags=["runtime-profile"])


def get_agent_runtime_profile_service(request: Request) -> AgentRuntimeProfileService:
    service = getattr(request.app.state, "agent_runtime_profile_service", None)
    if service is None:
        raise RuntimeError(
            "AgentRuntimeProfileService가 app.state에 설정되지 않았습니다."
        )
    return service


AgentRuntimeProfileServiceDep = Annotated[
    AgentRuntimeProfileService,
    Depends(get_agent_runtime_profile_service),
]


@router.get(
    "/current",
    response_model=AgentRuntimeProfilePayload,
    status_code=status.HTTP_200_OK,
)
def get_current_runtime_profile(
    service: AgentRuntimeProfileServiceDep,
) -> AgentRuntimeProfilePayload:
    try:
        return service.get_current_profile()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post(
    "/validate",
    response_model=AgentRuntimeProfileValidationResponsePayload,
    status_code=status.HTTP_200_OK,
)
def validate_runtime_profile(
    request: AgentRuntimeProfileValidationRequestPayload,
    service: AgentRuntimeProfileServiceDep,
) -> AgentRuntimeProfileValidationResponsePayload:
    try:
        return service.validate_profile(request)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
