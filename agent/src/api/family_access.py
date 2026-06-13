"""가족용 확장 프로그램의 setup/auth API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from agent.src.api.dependencies import FamilyAccessServiceDep
from agent.src.contracts.family_access_contracts import (
    FamilySetupRequestPayload,
    FamilySetupResponsePayload,
    FamilySetupStatusPayload,
    FamilyUnlockRequestPayload,
    FamilyUnlockResponsePayload,
)
from agent.src.features.wellbeing.family_access.service import (
    FamilyAccessSetupAlreadyCompletedError,
)

router = APIRouter(prefix="/api/v1/family", tags=["family-access"])


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
