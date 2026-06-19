"""동기화 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict

from agent.src.api.dependencies import (
    SharedAdapterRuntimeServiceDep,
    SharedAdapterSyncServiceDep,
)
from agent.src.infrastructure.repositories.shared_adapter_state_repository import (
    SharedAdapterStateActivationPointer,
)
from shared.src.contracts.adapter_contract_families.base import (
    CurrentSharedAdapterStatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_current_shared_adapter_state_payload,
)


class AssetPullRequest(BaseModel):
    """중앙 서버에서 현재 활성 asset을 내려받기 위한 요청."""

    model_config = ConfigDict(extra="forbid")

    server_base_url: str


router = APIRouter(prefix="/api/v1/sync", tags=["sync"])


@router.get(
    "/shared-adapters/current",
    response_model=CurrentSharedAdapterStatePayload,
)
def get_current_local_shared_adapter_state(
    runtime_service: SharedAdapterRuntimeServiceDep,
) -> CurrentSharedAdapterStatePayload:
    try:
        return make_current_shared_adapter_state_payload(
            manifest=runtime_service.get_active_manifest(),
            state=runtime_service.get_active_state(),
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post(
    "/shared-adapters/pull",
    response_model=SharedAdapterStateActivationPointer,
)
def pull_current_shared_adapter_state(
    request: AssetPullRequest,
    sync_service: SharedAdapterSyncServiceDep,
) -> SharedAdapterStateActivationPointer:
    try:
        return sync_service.pull_current(server_base_url=request.server_base_url)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"서버 통신 오류: {error}",
        ) from error
