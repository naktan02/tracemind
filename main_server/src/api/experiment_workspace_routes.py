"""experiment workspace route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from main_server.src.api.experiment_service_deps import (
    ExperimentWorkspaceServiceDep,
)
from main_server.src.services.experiments.payloads import (
    SavedWorkspaceDetailPayload,
    SavedWorkspaceSummaryPayload,
)
from shared.src.contracts.workspace_manifest_contracts import WorkspaceManifestPayload

router = APIRouter()


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
