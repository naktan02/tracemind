"""개발자용 experiment catalog API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from main_server.src.services.experiments.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiments.compiler_service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiments.payloads import (
    ExperimentCatalogPayload,
    ExperimentRunPayload,
    LaunchExperimentRunRequestPayload,
    SavedWorkspaceDetailPayload,
    SavedWorkspaceSummaryPayload,
)
from main_server.src.services.experiments.run_service import (
    ExperimentRunService,
)
from main_server.src.services.experiments.workspace_service import (
    ExperimentWorkspaceService,
)
from shared.src.contracts.workspace_manifest_contracts import (
    ResolvedExperimentPlanPayload,
    WorkspaceManifestPayload,
)

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


def get_experiment_catalog_service(request: Request) -> ExperimentCatalogService:
    """app.state에서 ExperimentCatalogService를 읽는다."""

    service = getattr(request.app.state, "experiment_catalog_service", None)
    if service is None:
        raise RuntimeError(
            "ExperimentCatalogService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.experiment_catalog_service를 설정하세요."
        )
    return service


ExperimentCatalogServiceDep = Annotated[
    ExperimentCatalogService,
    Depends(get_experiment_catalog_service),
]


def get_experiment_compiler_service(request: Request) -> ExperimentCompilerService:
    """app.state에서 ExperimentCompilerService를 읽는다."""

    service = getattr(request.app.state, "experiment_compiler_service", None)
    if service is None:
        raise RuntimeError(
            "ExperimentCompilerService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.experiment_compiler_service를 설정하세요."
        )
    return service


ExperimentCompilerServiceDep = Annotated[
    ExperimentCompilerService,
    Depends(get_experiment_compiler_service),
]


def get_experiment_workspace_service(request: Request) -> ExperimentWorkspaceService:
    """app.state에서 ExperimentWorkspaceService를 읽는다."""

    service = getattr(request.app.state, "experiment_workspace_service", None)
    if service is None:
        raise RuntimeError(
            "ExperimentWorkspaceService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.experiment_workspace_service를 설정하세요."
        )
    return service


ExperimentWorkspaceServiceDep = Annotated[
    ExperimentWorkspaceService,
    Depends(get_experiment_workspace_service),
]


def get_experiment_run_service(request: Request) -> ExperimentRunService:
    """app.state에서 ExperimentRunService를 읽는다."""

    service = getattr(request.app.state, "experiment_run_service", None)
    if service is None:
        raise RuntimeError(
            "ExperimentRunService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.experiment_run_service를 설정하세요."
        )
    return service


ExperimentRunServiceDep = Annotated[
    ExperimentRunService,
    Depends(get_experiment_run_service),
]


@router.get(
    "/catalog",
    response_model=ExperimentCatalogPayload,
)
def get_experiment_catalog(
    service: ExperimentCatalogServiceDep,
) -> ExperimentCatalogPayload:
    """현재 코드/설정 기준 read-only experiment catalog를 반환한다."""

    return service.build_catalog()


@router.post(
    "/compile",
    response_model=ResolvedExperimentPlanPayload,
)
def compile_experiment_manifest(
    manifest: WorkspaceManifestPayload,
    service: ExperimentCompilerServiceDep,
) -> ResolvedExperimentPlanPayload:
    """Workspace manifest를 기존 Hydra/script preview로 compile한다."""

    try:
        return service.compile_manifest(manifest)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


@router.get(
    "/workspaces",
    response_model=tuple[SavedWorkspaceSummaryPayload, ...],
)
def list_saved_experiment_workspaces(
    service: ExperimentWorkspaceServiceDep,
) -> tuple[SavedWorkspaceSummaryPayload, ...]:
    """저장된 experiment workspace 목록을 반환한다."""

    return service.list_workspaces()


@router.post(
    "/workspaces",
    response_model=SavedWorkspaceDetailPayload,
)
def save_experiment_workspace(
    manifest: WorkspaceManifestPayload,
    service: ExperimentWorkspaceServiceDep,
) -> SavedWorkspaceDetailPayload:
    """workspace manifest를 compile 후 저장한다."""

    try:
        return service.save_workspace(manifest)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


@router.get(
    "/workspaces/{workspace_id}",
    response_model=SavedWorkspaceDetailPayload,
)
def get_saved_experiment_workspace(
    workspace_id: str,
    service: ExperimentWorkspaceServiceDep,
) -> SavedWorkspaceDetailPayload:
    """단일 saved workspace detail을 반환한다."""

    try:
        return service.get_workspace(workspace_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


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
