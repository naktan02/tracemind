"""Captured text local ingest API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agent.src.api.dependencies import (
    CapturedTextLifecycleServiceDep,
    CapturedTextRepositoryDep,
    CapturedTextViewGenerationServiceDep,
    OptionalPipelineServiceDep,
    get_or_create_app_state,
)
from agent.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestRequestPayload,
    CapturedTextBatchIngestResponsePayload,
    CapturedTextDebugJobConfigRequestPayload,
    CapturedTextDebugJobRunRequestPayload,
    CapturedTextDebugJobRunResultPayload,
    CapturedTextDebugJobStatusPayload,
    CapturedTextEventPayload,
    CapturedTextIngestResponsePayload,
)
from agent.src.features.captured_text.debug_jobs import (
    CapturedTextDebugJobService,
)
from agent.src.features.captured_text.ingest import (
    CapturedTextIngestService,
)
from agent.src.features.captured_text.lifecycle import (
    CapturedTextLifecycleService,
)
from agent.src.features.captured_text.storage.repository import (
    CapturedTextRepository,
)
from agent.src.features.captured_text.view_generation.service import (
    CapturedTextViewGenerationService,
)
from agent.src.features.inference.pipeline_service import InferencePipelineService

router = APIRouter(prefix="/api/v1/captured-text", tags=["captured-text"])


@dataclass(slots=True)
class CapturedTextDebugJobState:
    """개발용 captured text background job runtime state."""

    enabled: bool = False
    interval_seconds: int = 30
    batch_size: int = 100
    last_run_at: datetime | None = None
    last_run_result: CapturedTextDebugJobRunResultPayload | None = None
    run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def get_captured_text_ingest_service(
    request: Request,
    repository: CapturedTextRepositoryDep,
    lifecycle_service: CapturedTextLifecycleServiceDep,
) -> CapturedTextIngestService:
    """app.state에서 captured text ingest service를 읽거나 조립한다."""

    return get_or_create_app_state(
        request,
        "captured_text_ingest_service",
        lambda: CapturedTextIngestService(
            captured_text_repository=repository,
            lifecycle_service=lifecycle_service,
        ),
    )


def get_captured_text_debug_job_state(request: Request) -> CapturedTextDebugJobState:
    """app.state에서 debug job state를 읽거나 생성한다."""

    return get_or_create_app_state(
        request,
        "captured_text_debug_job_state",
        CapturedTextDebugJobState,
    )


CapturedTextIngestServiceDep = Annotated[
    CapturedTextIngestService,
    Depends(get_captured_text_ingest_service),
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
    """단일 captured text event를 agent-local raw store에 저장한다."""

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
    """복수 captured text event를 agent-local raw store에 일괄 저장한다."""

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
        "analysis_status_counts": (
            service.captured_text_repository.count_by_analysis_status()
        ),
    }


@router.get(
    "/debug-job/status",
    response_model=CapturedTextDebugJobStatusPayload,
    status_code=status.HTTP_200_OK,
)
def captured_text_debug_job_status(
    request: Request,
    repository: CapturedTextRepositoryDep,
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
    repository: CapturedTextRepositoryDep,
    service: CapturedTextViewGenerationServiceDep,
    lifecycle_service: CapturedTextLifecycleServiceDep,
    job_state: CapturedTextDebugJobStateDep,
    pipeline_service: OptionalPipelineServiceDep,
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
                service=_debug_job_service(
                    repository=repository,
                    view_generation_service=service,
                    lifecycle_service=lifecycle_service,
                    pipeline_service=pipeline_service,
                ),
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
    request: Request,
    service: CapturedTextViewGenerationServiceDep,
    lifecycle_service: CapturedTextLifecycleServiceDep,
    job_state: CapturedTextDebugJobStateDep,
    pipeline_service: OptionalPipelineServiceDep,
) -> CapturedTextDebugJobRunResultPayload:
    """pending captured text view generation과 weak text 분석을 즉시 실행한다."""

    if job_state.run_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="captured text debug job이 이미 실행 중입니다.",
        )

    async with job_state.run_lock:
        result = await asyncio.to_thread(
            _debug_job_service(
                repository=service.repository,
                view_generation_service=service,
                lifecycle_service=lifecycle_service,
                pipeline_service=pipeline_service,
            ).run_once,
            limit=run_request.limit,
        )
        job_state.last_run_at = datetime.now(tz=timezone.utc)
        job_state.last_run_result = result
        return job_state.last_run_result


async def _captured_text_debug_job_loop(
    *,
    job_state: CapturedTextDebugJobState,
    service: CapturedTextDebugJobService,
) -> None:
    while job_state.enabled:
        async with job_state.run_lock:
            result = await asyncio.to_thread(
                service.run_once,
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
    is_running = (
        bool(task is not None and not task.done()) or job_state.run_lock.locked()
    )
    return CapturedTextDebugJobStatusPayload(
        view_generation_enabled=job_state.enabled,
        view_generation_running=is_running,
        view_generation_interval_seconds=job_state.interval_seconds,
        view_generation_batch_size=job_state.batch_size,
        weak_text_provider_name=service.weak_text_provider_name,
        strong_text_provider_name=service.strong_text_provider_name,
        weak_text_identity_fallback=service.weak_text_identity_fallback,
        strong_text_identity_fallback=service.strong_text_identity_fallback,
        captured_text_event_count=repository.count(),
        generated_view_count=repository.count_generated_views(),
        view_generation_status_counts=repository.count_by_view_generation_status(),
        analysis_status_counts=repository.count_by_analysis_status(),
        last_run_at=job_state.last_run_at,
        last_run_result=job_state.last_run_result,
    )


def _debug_job_service(
    *,
    repository: CapturedTextRepository,
    view_generation_service: CapturedTextViewGenerationService,
    lifecycle_service: CapturedTextLifecycleService,
    pipeline_service: InferencePipelineService | None,
) -> CapturedTextDebugJobService:
    return CapturedTextDebugJobService(
        repository=repository,
        view_generation_service=view_generation_service,
        lifecycle_service=lifecycle_service,
        pipeline_service=pipeline_service,
    )
