"""동기화 라우터."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from agent.src.features.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.features.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
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


def get_shared_adapter_runtime_service(request: Request) -> SharedAdapterRuntimeService:
    """app.state에서 SharedAdapterRuntimeService를 읽는다."""
    service = getattr(request.app.state, "shared_adapter_runtime_service", None)
    if service is None:
        raise RuntimeError(
            "SharedAdapterRuntimeService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.shared_adapter_runtime_service를 설정하세요."
        )
    return service


def get_shared_adapter_sync_service(request: Request) -> SharedAdapterSyncService:
    """app.state에서 SharedAdapterSyncService를 읽는다."""
    service = getattr(request.app.state, "shared_adapter_sync_service", None)
    if service is None:
        raise RuntimeError(
            "SharedAdapterSyncService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.shared_adapter_sync_service를 설정하세요."
        )
    return service


SharedAdapterRuntimeServiceDep = Annotated[
    SharedAdapterRuntimeService,
    Depends(get_shared_adapter_runtime_service),
]
SharedAdapterSyncServiceDep = Annotated[
    SharedAdapterSyncService,
    Depends(get_shared_adapter_sync_service),
]


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
