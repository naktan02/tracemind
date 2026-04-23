"""FL round lifecycle 라우터."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from main_server.src.services.federation.rounds.boundary.mappers import (
    round_finalize_request_from_payload,
    round_open_request_from_payload,
    round_record_to_payload,
    round_update_acceptance_to_payload,
    training_update_from_payload,
)
from main_server.src.services.federation.rounds.boundary.payloads import (
    RoundFinalizeRequestPayload,
    RoundOpenRequestPayload,
    RoundRecordPayload,
    RoundUpdateAcceptancePayload,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundConflictError,
    RoundLifecycleService,
    RoundValidationError,
)
from shared.src.contracts.training_contracts import TrainingUpdateEnvelopePayload

router = APIRouter(prefix="/api/v1/fl/rounds", tags=["fl-rounds"])


def get_round_lifecycle_service(request: Request) -> RoundLifecycleService:
    """RoundLifecycleService 의존성 provider.

    app.state에 등록된 서버 소유 인스턴스를 반환한다.
    테스트에서는 app.dependency_overrides[get_round_lifecycle_service]로
    격리된 인스턴스로 교체한다.
    """
    service = getattr(request.app.state, "round_lifecycle_service", None)
    if service is None:
        raise RuntimeError(
            "RoundLifecycleService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.round_lifecycle_service를 설정하세요."
        )
    return service


RoundServiceDep = Annotated[RoundLifecycleService, Depends(get_round_lifecycle_service)]


@router.get("/current", response_model=RoundRecordPayload)
def get_current_round(service: RoundServiceDep) -> RoundRecordPayload:
    try:
        return round_record_to_payload(service.get_current_round())
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post("", response_model=RoundRecordPayload, status_code=status.HTTP_201_CREATED)
def open_round(
    request: RoundOpenRequestPayload,
    service: RoundServiceDep,
) -> RoundRecordPayload:
    try:
        return round_record_to_payload(
            service.open_round(round_open_request_from_payload(request))
        )
    except RoundValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except RoundConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.get("/{round_id}", response_model=RoundRecordPayload)
def get_round(round_id: str, service: RoundServiceDep) -> RoundRecordPayload:
    try:
        return round_record_to_payload(service.get_round(round_id))
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post(
    "/{round_id}/updates",
    response_model=RoundUpdateAcceptancePayload,
    status_code=status.HTTP_202_ACCEPTED,
)
def accept_update(
    round_id: str,
    request: TrainingUpdateEnvelopePayload,
    service: RoundServiceDep,
) -> RoundUpdateAcceptancePayload:
    try:
        return round_update_acceptance_to_payload(
            service.accept_update(round_id, training_update_from_payload(request))
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except RoundValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except RoundConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.post("/{round_id}/finalize", response_model=RoundRecordPayload)
def finalize_round(
    round_id: str,
    request: RoundFinalizeRequestPayload,
    service: RoundServiceDep,
) -> RoundRecordPayload:
    try:
        return round_record_to_payload(
            service.finalize_round(
                round_id,
                round_finalize_request_from_payload(request),
            )
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except RoundValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except RoundConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
