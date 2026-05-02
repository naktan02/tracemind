"""개발자용 experiment API composition root."""

from __future__ import annotations

from fastapi import APIRouter

from main_server.src.api.experiment_catalog_routes import (
    compile_experiment_manifest,
    get_experiment_catalog,
)
from main_server.src.api.experiment_catalog_routes import (
    router as experiment_catalog_router,
)
from main_server.src.api.experiment_run_routes import (
    get_experiment_run,
    get_experiment_run_log,
    launch_experiment_run,
    list_experiment_runs,
)
from main_server.src.api.experiment_run_routes import (
    router as experiment_run_router,
)
from main_server.src.api.experiment_workspace_routes import (
    delete_saved_experiment_workspace,
    get_saved_experiment_workspace,
    list_saved_experiment_workspaces,
    save_experiment_workspace,
)
from main_server.src.api.experiment_workspace_routes import (
    router as experiment_workspace_router,
)

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])
router.include_router(experiment_catalog_router)
router.include_router(experiment_workspace_router)
router.include_router(experiment_run_router)

__all__ = [
    "compile_experiment_manifest",
    "delete_saved_experiment_workspace",
    "get_experiment_catalog",
    "get_experiment_run",
    "get_experiment_run_log",
    "get_saved_experiment_workspace",
    "launch_experiment_run",
    "list_experiment_runs",
    "list_saved_experiment_workspaces",
    "router",
    "save_experiment_workspace",
]
