"""experiment catalog/compile route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from main_server.src.api.experiment_service_deps import (
    ExperimentCatalogServiceDep,
    ExperimentCompilerServiceDep,
)
from main_server.src.services.experiments.payloads import ExperimentCatalogPayload
from shared.src.contracts.workspace_manifest_contracts import (
    ResolvedExperimentPlanPayload,
    WorkspaceManifestPayload,
)

router = APIRouter()


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
