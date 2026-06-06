"""Captured text local ingest API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agent.src.services.ingest.captured_text_ingest_service import (
    CapturedTextIngestService,
)
from shared.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestRequestPayload,
    CapturedTextBatchIngestResponsePayload,
    CapturedTextEventPayload,
    CapturedTextIngestResponsePayload,
)

router = APIRouter(prefix="/api/v1/captured-text", tags=["captured-text"])


def get_captured_text_ingest_service(
    request: Request,
) -> CapturedTextIngestService:
    """app.state에서 captured text ingest service를 읽거나 조립한다."""

    service = getattr(request.app.state, "captured_text_ingest_service", None)
    if service is not None:
        return service
    pipeline_service = getattr(request.app.state, "pipeline_service", None)
    captured_text_repository = getattr(
        request.app.state,
        "captured_text_repository",
        None,
    )
    if pipeline_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "CapturedTextIngestService를 만들 pipeline_service가 없습니다. "
                "앱 시작 시 app.state.pipeline_service를 설정하세요."
            ),
        )
    if captured_text_repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "CapturedTextIngestService를 만들 captured_text_repository가 "
                "없습니다. 앱 시작 시 app.state.captured_text_repository를 "
                "설정하세요."
            ),
        )
    service = CapturedTextIngestService(
        pipeline_service=pipeline_service,
        captured_text_repository=captured_text_repository,
    )
    request.app.state.captured_text_ingest_service = service
    return service


CapturedTextIngestServiceDep = Annotated[
    CapturedTextIngestService,
    Depends(get_captured_text_ingest_service),
]


@router.post(
    "/events",
    response_model=CapturedTextIngestResponsePayload,
    status_code=status.HTTP_201_CREATED,
)
def ingest_captured_text_event(
    request: CapturedTextEventPayload,
    service: CapturedTextIngestServiceDep,
) -> CapturedTextIngestResponsePayload:
    """단일 captured text event를 받아 agent-local inference pipeline을 실행한다."""

    try:
        return service.process(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"captured text event 처리 오류: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=CapturedTextBatchIngestResponsePayload,
    status_code=status.HTTP_201_CREATED,
)
def ingest_captured_text_batch(
    request: CapturedTextBatchIngestRequestPayload,
    service: CapturedTextIngestServiceDep,
) -> CapturedTextBatchIngestResponsePayload:
    """복수 captured text event를 일괄 처리한다."""

    try:
        return service.process_batch(request.events)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"captured text event batch 처리 오류: {exc}",
        ) from exc


@router.get("/status", status_code=status.HTTP_200_OK)
def captured_text_status(
    service: CapturedTextIngestServiceDep,
) -> dict[str, int]:
    """저장된 captured text event 수를 반환한다."""

    return {
        "captured_text_event_count": service.captured_text_repository.count(),
        "stored_event_count": service.pipeline_service.event_repository.count(),
    }
