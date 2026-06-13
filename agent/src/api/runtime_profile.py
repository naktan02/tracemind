"""Agent runtime profile debug/sync API."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from agent.src.api.dependencies import (
    RuntimeProfileRepositoryDep,
    RuntimeProfileSyncServiceDep,
)
from agent.src.features.runtime_profile.repository import RuntimeProfileRepository
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
)

router = APIRouter(prefix="/api/v1/runtime-profile", tags=["runtime-profile"])


class RuntimeProfileSyncRequest(BaseModel):
    """서버 active runtime profile sync 요청."""

    model_config = ConfigDict(extra="forbid")

    server_base_url: str = Field(min_length=1)


class RuntimeProfileStatusPayload(BaseModel):
    """agent-local active runtime profile 상태."""

    model_config = ConfigDict(extra="forbid")

    has_active_profile: bool
    profile: AgentRuntimeProfilePayload | None = None
    source: str | None = None
    received_at: datetime | None = None
    activated_at: datetime | None = None
    server_validated_at: datetime | None = None


class RuntimeProfileSyncResponse(BaseModel):
    """runtime profile sync 응답."""

    model_config = ConfigDict(extra="forbid")

    status: str
    message: str
    active_profile: RuntimeProfileStatusPayload


@router.get(
    "/status",
    response_model=RuntimeProfileStatusPayload,
    status_code=status.HTTP_200_OK,
)
def get_runtime_profile_status(
    repository: RuntimeProfileRepositoryDep,
) -> RuntimeProfileStatusPayload:
    return _profile_status(repository)


@router.post(
    "/sync",
    response_model=RuntimeProfileSyncResponse,
    status_code=status.HTTP_200_OK,
)
def sync_runtime_profile(
    request: RuntimeProfileSyncRequest,
    service: RuntimeProfileSyncServiceDep,
    repository: RuntimeProfileRepositoryDep,
) -> RuntimeProfileSyncResponse:
    try:
        result = service.sync_current(server_base_url=request.server_base_url)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"runtime profile sync failed: {error}",
        ) from error
    return RuntimeProfileSyncResponse(
        status=result.status,
        message=result.message,
        active_profile=_profile_status(repository),
    )


def _profile_status(
    repository: RuntimeProfileRepository,
) -> RuntimeProfileStatusPayload:
    record = repository.load_active()
    if record is None:
        return RuntimeProfileStatusPayload(has_active_profile=False)
    return RuntimeProfileStatusPayload(
        has_active_profile=True,
        profile=record.profile,
        source=record.source,
        received_at=record.received_at,
        activated_at=record.activated_at,
        server_validated_at=record.server_validated_at,
    )
