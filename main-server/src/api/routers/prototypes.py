"""PrototypePack 배포 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from shared.src.contracts.prototype_contracts import (
    CurrentPrototypePackResponse,
    PrototypePackActivationPointer,
    PrototypePackActivationRequest,
    PrototypePackPayload,
)
from src.services.prototypes.prototype_pack_service import PrototypePackService

router = APIRouter(prefix="/api/v1/prototypes", tags=["prototypes"])
service = PrototypePackService()


@router.get(
    "/current",
    response_model=CurrentPrototypePackResponse,
)
def get_current_prototype_pack() -> CurrentPrototypePackResponse:
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
def get_prototype_pack(prototype_version: str) -> PrototypePackPayload:
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
) -> PrototypePackActivationPointer:
    try:
        return service.activate(request.prototype_version)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
