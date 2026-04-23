"""experiment run route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from main_server.src.api.experiment_service_deps import ExperimentRunServiceDep
from main_server.src.services.experiments.payloads import (
    ExperimentRunPayload,
    LaunchExperimentRunRequestPayload,
)

router = APIRouter()


@router.get(
    "/runs",
    response_model=tuple[ExperimentRunPayload, ...],
)
def list_experiment_runs(
    service: ExperimentRunServiceDep,
) -> tuple[ExperimentRunPayload, ...]:
    """최근 local experiment run 목록을 반환한다."""

    return service.list_runs()


@router.post(
    "/runs",
    response_model=ExperimentRunPayload,
)
def launch_experiment_run(
    request: LaunchExperimentRunRequestPayload,
    service: ExperimentRunServiceDep,
) -> ExperimentRunPayload:
    """local experiment run을 시작한다."""

    try:
        return service.launch_run(request)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


@router.get(
    "/runs/{run_id}",
    response_model=ExperimentRunPayload,
)
def get_experiment_run(
    run_id: str,
    service: ExperimentRunServiceDep,
) -> ExperimentRunPayload:
    """단일 local experiment run 상태를 반환한다."""

    try:
        return service.get_run(run_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.get("/runs/{run_id}/logs/{stream_name}")
def get_experiment_run_log(
    run_id: str,
    stream_name: str,
    service: ExperimentRunServiceDep,
) -> FileResponse:
    """stdout/stderr log file을 다운로드 가능한 응답으로 반환한다."""

    try:
        log_path = service.read_log(run_id, stream_name=stream_name)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return FileResponse(
        log_path,
        media_type="text/plain",
        filename=log_path.name,
    )
