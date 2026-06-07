"""Typing segment local ingest API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agent.src.contracts.typing_segment_contracts import (
    TypingSegmentBatchIngestRequestPayload,
    TypingSegmentBatchIngestResponsePayload,
    TypingSegmentIngestResponsePayload,
    TypingSegmentPayload,
)
from agent.src.services.typing_segments.ingest_service import (
    TypingSegmentIngestService,
)

router = APIRouter(prefix="/api/v1/typing-segments", tags=["typing-segments"])


def get_typing_segment_ingest_service(
    request: Request,
) -> TypingSegmentIngestService:
    """app.state에서 typing segment ingest service를 읽거나 조립한다."""

    service = getattr(request.app.state, "typing_segment_ingest_service", None)
    if service is not None:
        return service
    pipeline_service = getattr(request.app.state, "pipeline_service", None)
    if pipeline_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "TypingSegmentIngestService를 만들 pipeline_service가 없습니다. "
                "앱 시작 시 app.state.pipeline_service를 설정하세요."
            ),
        )
    service = TypingSegmentIngestService(pipeline_service=pipeline_service)
    request.app.state.typing_segment_ingest_service = service
    return service


TypingSegmentIngestServiceDep = Annotated[
    TypingSegmentIngestService,
    Depends(get_typing_segment_ingest_service),
]


@router.post(
    "",
    response_model=TypingSegmentIngestResponsePayload,
    status_code=status.HTTP_201_CREATED,
)
def ingest_typing_segment(
    request: TypingSegmentPayload,
    service: TypingSegmentIngestServiceDep,
) -> TypingSegmentIngestResponsePayload:
    """단일 typing segment를 받아 agent-local inference pipeline을 실행한다."""

    try:
        return service.process(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"typing segment 처리 오류: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=TypingSegmentBatchIngestResponsePayload,
    status_code=status.HTTP_201_CREATED,
)
def ingest_typing_segment_batch(
    request: TypingSegmentBatchIngestRequestPayload,
    service: TypingSegmentIngestServiceDep,
) -> TypingSegmentBatchIngestResponsePayload:
    """복수 typing segment를 일괄 처리한다."""

    try:
        return service.process_batch(request.segments)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"typing segment batch 처리 오류: {exc}",
        ) from exc
