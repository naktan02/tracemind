"""개발자용 experiment API composition root."""

from __future__ import annotations

from fastapi import APIRouter

from main_server.src.api.experiment_catalog_routes import (
    compile_experiment_manifest as compile_experiment_manifest,
)
from main_server.src.api.experiment_catalog_routes import (
    get_experiment_catalog as get_experiment_catalog,
)
from main_server.src.api.experiment_catalog_routes import (
    router as experiment_catalog_router,
)
from main_server.src.api.experiment_run_routes import (
    get_experiment_run as get_experiment_run,
)
from main_server.src.api.experiment_run_routes import (
    get_experiment_run_log as get_experiment_run_log,
)
from main_server.src.api.experiment_run_routes import (
    launch_experiment_run as launch_experiment_run,
)
from main_server.src.api.experiment_run_routes import (
    list_experiment_runs as list_experiment_runs,
)
from main_server.src.api.experiment_run_routes import (
    router as experiment_run_router,
)
from main_server.src.api.experiment_workspace_routes import (
    delete_saved_experiment_workspace as delete_saved_experiment_workspace,
)
from main_server.src.api.experiment_workspace_routes import (
    get_saved_experiment_workspace as get_saved_experiment_workspace,
)
from main_server.src.api.experiment_workspace_routes import (
    list_saved_experiment_workspaces as list_saved_experiment_workspaces,
)
from main_server.src.api.experiment_workspace_routes import (
    router as experiment_workspace_router,
)
from main_server.src.api.experiment_workspace_routes import (
    save_experiment_workspace as save_experiment_workspace,
)

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])
router.include_router(experiment_catalog_router)
router.include_router(experiment_workspace_router)
router.include_router(experiment_run_router)
