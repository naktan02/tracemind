"""동기화 라우터."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from agent.src.infrastructure.repositories.shared_adapter_state_repository import (
    SharedAdapterStateActivationPointer,
)
from agent.src.services.assets.prototypes.runtime_service import PrototypeRuntimeService
from agent.src.services.assets.prototypes.sync_service import PrototypeSyncService
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from shared.src.contracts.adapter_contracts import (
    CurrentSharedAdapterStatePayload,
    make_current_shared_adapter_state_payload,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackActivationPointer,
    PrototypePackPayload,
)


class PrototypePullRequest(BaseModel):
    """중앙 서버에서 현재 활성 prototype pack을 내려받기 위한 요청."""

    model_config = ConfigDict(extra="forbid")

    server_base_url: str


router = APIRouter(prefix="/api/v1/sync", tags=["sync"])


def get_prototype_runtime_service(request: Request) -> PrototypeRuntimeService:
    """app.state에서 PrototypeRuntimeService를 읽는다."""
    service = getattr(request.app.state, "prototype_runtime_service", None)
    if service is None:
        raise RuntimeError(
            "PrototypeRuntimeService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.prototype_runtime_service를 설정하세요."
        )
    return service


def get_prototype_sync_service(request: Request) -> PrototypeSyncService:
    """app.state에서 PrototypeSyncService를 읽는다."""
    service = getattr(request.app.state, "prototype_sync_service", None)
    if service is None:
        raise RuntimeError(
            "PrototypeSyncService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.prototype_sync_service를 설정하세요."
        )
    return service


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


PrototypeRuntimeServiceDep = Annotated[
    PrototypeRuntimeService,
    Depends(get_prototype_runtime_service),
]
PrototypeSyncServiceDep = Annotated[
    PrototypeSyncService,
    Depends(get_prototype_sync_service),
]
SharedAdapterRuntimeServiceDep = Annotated[
    SharedAdapterRuntimeService,
    Depends(get_shared_adapter_runtime_service),
]
SharedAdapterSyncServiceDep = Annotated[
    SharedAdapterSyncService,
    Depends(get_shared_adapter_sync_service),
]


@router.get(
    "/prototypes/current",
    response_model=PrototypePackPayload,
)
def get_current_local_prototype_pack(
    runtime_service: PrototypeRuntimeServiceDep,
) -> PrototypePackPayload:
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
    sync_service: PrototypeSyncServiceDep,
) -> PrototypePackActivationPointer:
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
    request: PrototypePullRequest,
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
