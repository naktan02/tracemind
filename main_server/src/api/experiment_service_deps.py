"""experiment API service dependency helpers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from main_server.src.services.experiment_workspace.catalog.service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.compiler.service import (
    ExperimentCompilerService,
)
from main_server.src.services.experiment_workspace.run_execution.service import (
    ExperimentRunService,
)
from main_server.src.services.experiment_workspace.workspace_service import (
    ExperimentWorkspaceService,
)


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
