"""PrototypePack 배포 라우터."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from main_server.src.services.prototype_assets.prototype_pack_service import (
    PrototypePackService,
)
from shared.src.contracts.prototype_contracts import (
    CurrentPrototypePackResponse,
    PrototypePackActivationPointer,
    PrototypePackActivationRequest,
    PrototypePackPayload,
)

router = APIRouter(prefix="/api/v1/prototypes", tags=["prototypes"])


def get_prototype_pack_service(request: Request) -> PrototypePackService:
    """app.state에서 PrototypePackService를 읽는다."""
    service = getattr(request.app.state, "prototype_pack_service", None)
    if service is None:
        raise RuntimeError(
            "PrototypePackService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.prototype_pack_service를 설정하세요."
        )
    return service


PrototypePackServiceDep = Annotated[
    PrototypePackService,
    Depends(get_prototype_pack_service),
]


@router.get(
    "/current",
    response_model=CurrentPrototypePackResponse,
)
def get_current_prototype_pack(
    service: PrototypePackServiceDep,
) -> CurrentPrototypePackResponse:
    try:
        return service.get_current()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.get(
    "/{prototype_version}",
    response_model=PrototypePackPayload,
)
def get_prototype_pack(
    prototype_version: str,
    service: PrototypePackServiceDep,
) -> PrototypePackPayload:
    try:
        return service.get_pack(prototype_version)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post(
    "/activate",
    response_model=PrototypePackActivationPointer,
)
def activate_prototype_pack(
    request: PrototypePackActivationRequest,
    service: PrototypePackServiceDep,
) -> PrototypePackActivationPointer:
    try:
        return service.activate(request.prototype_version)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
