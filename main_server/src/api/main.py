"""중앙 서버 API 진입점."""

from __future__ import annotations

import os
from collections.abc import Mapping

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main_server.src.api.experiments import router as experiments_router
from main_server.src.api.fl_rounds import router as fl_rounds_router
from main_server.src.api.health import router as health_router
from main_server.src.api.prototypes import router as prototypes_router
from main_server.src.services.experiments.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiments.compiler_service import (
    ExperimentCompilerService,
)
from main_server.src.services.prototypes.prototype_pack_service import (
    PrototypePackService,
)
from main_server.src.services.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.rounds.runtime_config import (
    ServerRoundRuntimeConfig,
    load_server_round_runtime_config_from_env,
)
from main_server.src.services.rounds.runtime_factory import (
    build_round_lifecycle_service_from_config,
)

EXPERIMENT_WEB_ALLOWED_ORIGINS_ENV = "EXPERIMENT_WEB_ALLOWED_ORIGINS"
DEFAULT_EXPERIMENT_WEB_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def load_experiment_web_allowed_origins_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """experiment web dev server가 접근할 수 있는 origin 목록을 읽는다."""

    effective_environ = os.environ if environ is None else environ
    raw_value = effective_environ.get(EXPERIMENT_WEB_ALLOWED_ORIGINS_ENV, "")
    origins = tuple(
        origin.strip() for origin in raw_value.split(",") if origin.strip()
    )
    return origins or DEFAULT_EXPERIMENT_WEB_ALLOWED_ORIGINS


def create_app(
    *,
    round_lifecycle_service: RoundLifecycleService | None = None,
    round_runtime_config: ServerRoundRuntimeConfig | None = None,
    prototype_pack_service: PrototypePackService | None = None,
    experiment_catalog_service: ExperimentCatalogService | None = None,
    experiment_compiler_service: ExperimentCompilerService | None = None,
    experiment_web_allowed_origins: tuple[str, ...] | None = None,
) -> FastAPI:
    """Main server 앱을 생성하고 서버 소유 서비스를 app.state에 연결한다."""
    app = FastAPI(title="TraceMind Main Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            experiment_web_allowed_origins
            or load_experiment_web_allowed_origins_from_env()
        ),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    effective_runtime_config = (
        round_runtime_config or load_server_round_runtime_config_from_env()
    )
    app.state.round_runtime_config = effective_runtime_config
    app.state.round_lifecycle_service = (
        round_lifecycle_service
        or build_round_lifecycle_service_from_config(effective_runtime_config)
    )
    app.state.prototype_pack_service = prototype_pack_service or PrototypePackService()
    effective_experiment_catalog_service = (
        experiment_catalog_service or ExperimentCatalogService()
    )
    app.state.experiment_catalog_service = effective_experiment_catalog_service
    app.state.experiment_compiler_service = (
        experiment_compiler_service
        or ExperimentCompilerService(
            catalog_service=effective_experiment_catalog_service
        )
    )
    app.include_router(health_router)
    app.include_router(experiments_router)
    app.include_router(fl_rounds_router)
    app.include_router(prototypes_router)
    return app


app = create_app()
