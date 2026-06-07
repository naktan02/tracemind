"""가족용 확장 프로그램의 setup/auth API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agent.src.contracts.family_access_contracts import (
    FamilySetupRequestPayload,
    FamilySetupResponsePayload,
    FamilySetupStatusPayload,
    FamilyUnlockRequestPayload,
    FamilyUnlockResponsePayload,
)
from agent.src.services.wellbeing.family_access_service import (
    FamilyAccessService,
    FamilyAccessSetupAlreadyCompletedError,
)

router = APIRouter(prefix="/api/v1/family", tags=["family-access"])


def get_family_access_service(request: Request) -> FamilyAccessService:
    service = getattr(request.app.state, "family_access_service", None)
    if service is None:
        raise RuntimeError(
            "FamilyAccessService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.family_access_service를 설정하세요."
        )
    return service


FamilyAccessServiceDep = Annotated[
    FamilyAccessService,
    Depends(get_family_access_service),
]


@router.get(
    "/setup/status",
    response_model=FamilySetupStatusPayload,
)
def get_family_setup_status(
    family_access_service: FamilyAccessServiceDep,
) -> FamilySetupStatusPayload:
    """초기 setup 완료 여부와 로컬 전용 mode 상태를 반환한다."""

    return family_access_service.get_setup_status()


@router.post(
    "/setup",
    response_model=FamilySetupResponsePayload,
)
def create_family_setup(
    request: FamilySetupRequestPayload,
    family_access_service: FamilyAccessServiceDep,
) -> FamilySetupResponsePayload:
    """child/parent PIN을 함께 설정하는 최초 setup."""

    try:
        return family_access_service.create_initial_setup(
            child_pin=request.child_pin,
            parent_pin=request.parent_pin,
        )
    except FamilyAccessSetupAlreadyCompletedError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.post(
    "/unlock",
    response_model=FamilyUnlockResponsePayload,
)
def unlock_family_role(
    request: FamilyUnlockRequestPayload,
    family_access_service: FamilyAccessServiceDep,
) -> FamilyUnlockResponsePayload:
    """role별 PIN 잠금 해제."""

    return family_access_service.unlock(role=request.role, pin=request.pin)
