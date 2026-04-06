"""중앙 서버 API 진입점."""

from __future__ import annotations

from fastapi import FastAPI

from main_server.src.api.fl_rounds import router as fl_rounds_router
from main_server.src.api.health import router as health_router
from main_server.src.api.prototypes import router as prototypes_router
from main_server.src.services.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)


def create_app(
    *,
    round_lifecycle_service: RoundLifecycleService | None = None,
) -> FastAPI:
    """Main server 앱을 생성하고 서버 소유 서비스를 app.state에 연결한다."""
    app = FastAPI(title="TraceMind Main Server", version="0.1.0")
    app.state.round_lifecycle_service = (
        round_lifecycle_service or RoundLifecycleService()
    )
    app.include_router(health_router)
    app.include_router(fl_rounds_router)
    app.include_router(prototypes_router)
    return app


app = create_app()
