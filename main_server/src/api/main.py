"""중앙 서버 API 진입점."""

from __future__ import annotations

from fastapi import FastAPI

from main_server.src.api.admin import router as admin_router
from main_server.src.api.agent_runtime_profile import (
    router as agent_runtime_profile_router,
)
from main_server.src.api.fl_rounds import router as fl_rounds_router
from main_server.src.api.health import router as health_router
from main_server.src.services.agent_runtime_profile_service import (
    AgentRuntimeProfileService,
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
from main_server.src.services.federation.strategy.active_strategy_service import (
    ActiveStrategyService,
)


def create_app(
    *,
    round_lifecycle_service: RoundLifecycleService | None = None,
    round_runtime_config: ServerRoundRuntimeConfig | None = None,
    active_strategy_service: ActiveStrategyService | None = None,
    agent_runtime_profile_service: AgentRuntimeProfileService | None = None,
) -> FastAPI:
    """Main server 앱을 생성하고 서버 소유 서비스를 app.state에 연결한다."""
    app = FastAPI(title="TraceMind Main Server", version="0.1.0")
    effective_runtime_config = (
        round_runtime_config or load_server_round_runtime_config_from_env()
    )
    effective_strategy_service = active_strategy_service or ActiveStrategyService()
    app.state.round_runtime_config = effective_runtime_config
    app.state.active_strategy_service = effective_strategy_service
    effective_round_lifecycle_service = (
        round_lifecycle_service
        or build_round_lifecycle_service_from_config(
            effective_runtime_config,
            active_strategy_service=effective_strategy_service,
        )
    )
    app.state.round_lifecycle_service = effective_round_lifecycle_service
    app.state.agent_runtime_profile_service = (
        agent_runtime_profile_service
        or AgentRuntimeProfileService(
            active_manifest_service=(
                effective_round_lifecycle_service.active_manifest_service
            )
        )
    )
    app.include_router(health_router)
    app.include_router(fl_rounds_router)
    app.include_router(agent_runtime_profile_router)
    app.include_router(admin_router)
    return app


app = create_app()
