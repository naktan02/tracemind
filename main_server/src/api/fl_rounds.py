"""FL round lifecycle 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from main_server.src.services.rounds.models import (
    RoundFinalizeRequestPayload,
    RoundOpenRequestPayload,
    RoundRecordPayload,
    RoundUpdateAcceptancePayload,
    round_finalize_request_from_payload,
    round_open_request_from_payload,
    round_record_to_payload,
    round_update_acceptance_to_payload,
    training_update_from_payload,
)
from main_server.src.services.rounds.round_lifecycle_service import (
    RoundConflictError,
    RoundLifecycleService,
    RoundValidationError,
)
from shared.src.contracts.training_contracts import TrainingUpdateEnvelopePayload

router = APIRouter(prefix="/api/v1/fl/rounds", tags=["fl-rounds"])
service = RoundLifecycleService()


@router.get("/current", response_model=RoundRecordPayload)
def get_current_round() -> RoundRecordPayload:
    try:
        return round_record_to_payload(service.get_current_round())
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post("", response_model=RoundRecordPayload, status_code=status.HTTP_201_CREATED)
def open_round(request: RoundOpenRequestPayload) -> RoundRecordPayload:
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
def get_round(round_id: str) -> RoundRecordPayload:
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
