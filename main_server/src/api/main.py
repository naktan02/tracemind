"""중앙 서버 API 진입점."""

from __future__ import annotations

from fastapi import FastAPI

from main_server.src.api.fl_rounds import router as fl_rounds_router
from main_server.src.api.health import router as health_router
from main_server.src.api.prototypes import router as prototypes_router
from main_server.src.services.federation.prototypes import (
    prototype_pack_service as prototype_pack_service_module,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.federation.rounds.runtime.config import (
    ServerRoundRuntimeConfig,
    load_server_round_runtime_config_from_env,
)
from main_server.src.services.federation.rounds.runtime.factory import (
    build_round_lifecycle_service_from_config,
)

PrototypePackService = prototype_pack_service_module.PrototypePackService


def create_app(
    *,
    round_lifecycle_service: RoundLifecycleService | None = None,
    round_runtime_config: ServerRoundRuntimeConfig | None = None,
    prototype_pack_service: PrototypePackService | None = None,
) -> FastAPI:
    """Main server 앱을 생성하고 서버 소유 서비스를 app.state에 연결한다."""
    app = FastAPI(title="TraceMind Main Server", version="0.1.0")
    effective_runtime_config = (
        round_runtime_config or load_server_round_runtime_config_from_env()
    )
    app.state.round_runtime_config = effective_runtime_config
    app.state.round_lifecycle_service = (
        round_lifecycle_service
        or build_round_lifecycle_service_from_config(effective_runtime_config)
    )
    app.state.prototype_pack_service = prototype_pack_service or PrototypePackService()
    app.include_router(health_router)
    app.include_router(fl_rounds_router)
    app.include_router(prototypes_router)
    return app


app = create_app()
