"""입력 수집 라우터.

외부 텍스트 이벤트를 받아 inference pipeline을 실행하고 결과를 저장한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from agent.src.features.inference.pipeline_service import (
    InferencePipelineService,
)
from shared.src.domain.entities.inference.events import QueryEvent

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


# ------------------------------------------------------------------ #
# 요청/응답 모델                                                        #
# ------------------------------------------------------------------ #


class IngestEventRequest(BaseModel):
    """단일 텍스트 이벤트 수집 요청."""

    text: str = Field(min_length=1, description="원문 텍스트.")
    locale: str = Field(default="ko", description="텍스트 언어 코드. 예: ko, en, ja.")
    source_type: str = Field(default="manual", description="이벤트 발생 출처.")
    occurred_at: datetime | None = Field(
        default=None,
        description="이벤트 발생 시각 (UTC). 없으면 수신 시각으로 설정.",
    )


class IngestEventResponse(BaseModel):
    """이벤트 수집 결과."""

    query_id: str
    was_translated: bool
    top_category: str | None
    top_score: float | None
    message: str


class IngestBatchRequest(BaseModel):
    """복수 이벤트 일괄 수집 요청."""

    events: list[IngestEventRequest] = Field(min_length=1, max_length=100)


class IngestBatchResponse(BaseModel):
    """일괄 수집 결과."""

    processed: int
    results: list[IngestEventResponse]


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# 의존성 주입                                                            #
# ------------------------------------------------------------------ #


def get_pipeline_service(request: Request) -> InferencePipelineService:
    """app.state에서 InferencePipelineService를 읽는다.

    서비스는 앱 시작 시 main.py의 lifespan 또는 startup 이벤트에서
    app.state.pipeline_service = InferencePipelineService(...) 로 설정한다.

    테스트에서는 app.dependency_overrides[get_pipeline_service]로 교체한다.
    """
    service = getattr(request.app.state, "pipeline_service", None)
    if service is None:
        raise RuntimeError(
            "InferencePipelineService가 app.state에 설정되지 않았습니다. "
            "앱 startup 시 app.state.pipeline_service를 설정하세요."
        )
    return service


PipelineDep = Annotated[InferencePipelineService, Depends(get_pipeline_service)]


# ------------------------------------------------------------------ #
# 엔드포인트                                                            #
# ------------------------------------------------------------------ #


@router.post(
    "/event",
    response_model=IngestEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_event(
    request: IngestEventRequest,
    pipeline: PipelineDep,
) -> IngestEventResponse:
    """단일 텍스트 이벤트를 받아 inference pipeline을 실행한다."""
    event = QueryEvent(
        query_id=_new_id(),
        text=request.text,
        occurred_at=request.occurred_at or datetime.now(tz=timezone.utc),
        locale=request.locale,
        source_type=request.source_type,
    )
    try:
        result = pipeline.process(event)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파이프라인 실행 오류: {exc}",
        ) from exc

    scores = result.analysis_event.category_scores
    top_category, top_score = _top_score(scores)
    return IngestEventResponse(
        query_id=result.analysis_event.query_id,
        was_translated=result.was_translated,
        top_category=top_category,
        top_score=top_score,
        message="이벤트가 처리되어 저장되었습니다.",
    )


@router.post(
    "/batch",
    response_model=IngestBatchResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_batch(
    request: IngestBatchRequest,
    pipeline: PipelineDep,
) -> IngestBatchResponse:
    """복수 이벤트를 일괄 처리한다."""
    events = [
        QueryEvent(
            query_id=_new_id(),
            text=item.text,
            occurred_at=item.occurred_at or datetime.now(tz=timezone.utc),
            locale=item.locale,
            source_type=item.source_type,
        )
        for item in request.events
    ]
    try:
        results = pipeline.process_batch(events)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파이프라인 실행 오류: {exc}",
        ) from exc

    responses = []
    for result in results:
        scores = result.analysis_event.category_scores
        top_category, top_score = _top_score(scores)
        responses.append(
            IngestEventResponse(
                query_id=result.analysis_event.query_id,
                was_translated=result.was_translated,
                top_category=top_category,
                top_score=top_score,
                message="처리 완료.",
            )
        )
    return IngestBatchResponse(processed=len(responses), results=responses)


@router.get("/status", status_code=status.HTTP_200_OK)
def ingest_status(pipeline: PipelineDep) -> dict:
    """저장된 이벤트 수를 반환한다."""
    return {"stored_event_count": pipeline.event_repository.count()}


# ------------------------------------------------------------------ #
# 내부 유틸                                                             #
# ------------------------------------------------------------------ #


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())


def _top_score(scores: dict[str, float]) -> tuple[str | None, float | None]:
    if not scores:
        return None, None
    top = max(scores, key=scores.__getitem__)
    return top, scores[top]
