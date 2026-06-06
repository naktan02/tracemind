"""Captured text local ingest API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
)
from agent.src.services.ingest.captured_text_ingest_service import (
    CapturedTextIngestService,
)
from agent.src.services.ingest.captured_text_view_generation_service import (
    CapturedTextViewGenerationService,
)
from shared.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestRequestPayload,
    CapturedTextBatchIngestResponsePayload,
    CapturedTextDebugJobConfigRequestPayload,
    CapturedTextDebugJobRunRequestPayload,
    CapturedTextDebugJobRunResultPayload,
    CapturedTextDebugJobStatusPayload,
    CapturedTextEventPayload,
    CapturedTextIngestResponsePayload,
)

router = APIRouter(prefix="/api/v1/captured-text", tags=["captured-text"])


@dataclass(slots=True)
class CapturedTextDebugJobState:
    """개발용 captured text background job runtime state."""

    enabled: bool = False
    interval_seconds: int = 30
    batch_size: int = 100
    last_run_at: datetime | None = None
    last_run_result: CapturedTextDebugJobRunResultPayload | None = None


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


def get_captured_text_repository(request: Request) -> CapturedTextRepository:
    """app.state에서 captured text repository를 읽는다."""

    repository = getattr(request.app.state, "captured_text_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "captured_text_repository가 없습니다. 앱 시작 시 "
                "app.state.captured_text_repository를 설정하세요."
            ),
        )
    return repository


def get_captured_text_view_generation_service(
    request: Request,
) -> CapturedTextViewGenerationService:
    """app.state에서 view generation service를 읽거나 조립한다."""

    service = getattr(request.app.state, "captured_text_view_generation_service", None)
    if service is not None:
        return service
    repository = get_captured_text_repository(request)
    service = CapturedTextViewGenerationService(
        repository=repository,
        translation_provider=getattr(
            request.app.state,
            "captured_text_translation_service",
            None,
        ),
        strong_view_provider=getattr(
            request.app.state,
            "captured_text_strong_view_service",
            None,
        ),
    )
    request.app.state.captured_text_view_generation_service = service
    return service


def get_captured_text_debug_job_state(request: Request) -> CapturedTextDebugJobState:
    """app.state에서 debug job state를 읽거나 생성한다."""

    job_state = getattr(request.app.state, "captured_text_debug_job_state", None)
    if job_state is None:
        job_state = CapturedTextDebugJobState()
        request.app.state.captured_text_debug_job_state = job_state
    return job_state


CapturedTextIngestServiceDep = Annotated[
    CapturedTextIngestService,
    Depends(get_captured_text_ingest_service),
]
CapturedTextRepoDep = Annotated[
    CapturedTextRepository,
    Depends(get_captured_text_repository),
]
CapturedTextViewGenerationServiceDep = Annotated[
    CapturedTextViewGenerationService,
    Depends(get_captured_text_view_generation_service),
]
CapturedTextDebugJobStateDep = Annotated[
    CapturedTextDebugJobState,
    Depends(get_captured_text_debug_job_state),
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
) -> dict[str, object]:
    """저장된 captured text event 수를 반환한다."""

    return {
        "captured_text_event_count": service.captured_text_repository.count(),
        "view_generation_status_counts": (
            service.captured_text_repository.count_by_view_generation_status()
        ),
        "stored_event_count": service.pipeline_service.event_repository.count(),
    }


@router.get(
    "/debug-job/status",
    response_model=CapturedTextDebugJobStatusPayload,
    status_code=status.HTTP_200_OK,
)
def captured_text_debug_job_status(
    request: Request,
    repository: CapturedTextRepoDep,
    service: CapturedTextViewGenerationServiceDep,
    job_state: CapturedTextDebugJobStateDep,
) -> CapturedTextDebugJobStatusPayload:
    """debug page가 읽는 captured text pipeline 상태."""

    return _build_debug_job_status(
        request=request,
        repository=repository,
        service=service,
        job_state=job_state,
    )


@router.post(
    "/debug-job/config",
    response_model=CapturedTextDebugJobStatusPayload,
    status_code=status.HTTP_200_OK,
)
async def configure_captured_text_debug_job(
    config: CapturedTextDebugJobConfigRequestPayload,
    request: Request,
    repository: CapturedTextRepoDep,
    service: CapturedTextViewGenerationServiceDep,
    job_state: CapturedTextDebugJobStateDep,
) -> CapturedTextDebugJobStatusPayload:
    """개발용 captured text view generation job을 켜거나 끈다."""

    job_state.enabled = config.view_generation_enabled
    job_state.interval_seconds = config.view_generation_interval_seconds
    job_state.batch_size = config.view_generation_batch_size
    task = getattr(request.app.state, "captured_text_debug_job_task", None)
    if job_state.enabled and (task is None or task.done()):
        request.app.state.captured_text_debug_job_task = asyncio.create_task(
            _captured_text_debug_job_loop(
                job_state=job_state,
                service=service,
            )
        )
    if not job_state.enabled and task is not None:
        task.cancel()
        request.app.state.captured_text_debug_job_task = None
    return _build_debug_job_status(
        request=request,
        repository=repository,
        service=service,
        job_state=job_state,
    )


@router.post(
    "/debug-job/run-view-generation",
    response_model=CapturedTextDebugJobRunResultPayload,
    status_code=status.HTTP_200_OK,
)
async def run_captured_text_view_generation_once(
    run_request: CapturedTextDebugJobRunRequestPayload,
    service: CapturedTextViewGenerationServiceDep,
    job_state: CapturedTextDebugJobStateDep,
) -> CapturedTextDebugJobRunResultPayload:
    """pending captured text view generation을 즉시 한 번 실행한다."""

    result = await asyncio.to_thread(
        service.generate_pending_views,
        limit=run_request.limit,
    )
    job_state.last_run_at = datetime.now(tz=timezone.utc)
    job_state.last_run_result = result
    return result


async def _captured_text_debug_job_loop(
    *,
    job_state: CapturedTextDebugJobState,
    service: CapturedTextViewGenerationService,
) -> None:
    while job_state.enabled:
        result = await asyncio.to_thread(
            service.generate_pending_views,
            limit=job_state.batch_size,
        )
        job_state.last_run_at = datetime.now(tz=timezone.utc)
        job_state.last_run_result = result
        await asyncio.sleep(job_state.interval_seconds)


def _build_debug_job_status(
    *,
    request: Request,
    repository: CapturedTextRepository,
    service: CapturedTextViewGenerationService,
    job_state: CapturedTextDebugJobState,
) -> CapturedTextDebugJobStatusPayload:
    task = getattr(request.app.state, "captured_text_debug_job_task", None)
    return CapturedTextDebugJobStatusPayload(
        view_generation_enabled=job_state.enabled,
        view_generation_running=bool(task is not None and not task.done()),
        view_generation_interval_seconds=job_state.interval_seconds,
        view_generation_batch_size=job_state.batch_size,
        weak_text_provider_name=service.weak_text_provider_name,
        strong_text_provider_name=service.strong_text_provider_name,
        weak_text_identity_fallback=service.weak_text_identity_fallback,
        strong_text_identity_fallback=service.strong_text_identity_fallback,
        captured_text_event_count=repository.count(),
        generated_view_count=repository.count_generated_views(),
        view_generation_status_counts=repository.count_by_view_generation_status(),
        last_run_at=job_state.last_run_at,
        last_run_result=job_state.last_run_result,
    )
