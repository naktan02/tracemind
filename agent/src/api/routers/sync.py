"""동기화 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict

from agent.src.services.prototype_runtime_service import PrototypeRuntimeService
from agent.src.services.prototype_sync_service import PrototypeSyncService
from shared.src.contracts.prototype_contracts import (
    PrototypePackActivationPointer,
    PrototypePackPayload,
)


class PrototypePullRequest(BaseModel):
    """중앙 서버에서 현재 활성 prototype pack을 내려받기 위한 요청."""

    model_config = ConfigDict(extra="forbid")

    server_base_url: str


router = APIRouter(prefix="/api/v1/sync", tags=["sync"])
runtime_service = PrototypeRuntimeService()
sync_service = PrototypeSyncService()


@router.get(
    "/prototypes/current",
    response_model=PrototypePackPayload,
)
def get_current_local_prototype_pack() -> PrototypePackPayload:
    try:
        return runtime_service.get_active_pack()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post(
    "/prototypes/pull",
    response_model=PrototypePackActivationPointer,
)
def pull_current_prototype_pack(
    request: PrototypePullRequest,
) -> PrototypePackActivationPointer:
    try:
        return sync_service.pull_current(server_base_url=request.server_base_url)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
