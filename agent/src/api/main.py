"""Agent API 진입점."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from agent.src.api.health import router as health_router
from agent.src.api.ingest import router as ingest_router
from agent.src.api.sync import router as sync_router
from agent.src.api.training import router as training_router
from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.services.federation.round_client import RoundClient
from agent.src.services.federation.runtime_service import FederationRuntimeService
from agent.src.services.inference.pipeline_service import InferencePipelineService
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.prototype.runtime_service import PrototypeRuntimeService

RoundClientFactory = Callable[[str], RoundClient]
FederationRuntimeServiceFactory = Callable[[str], FederationRuntimeService]


def _default_round_client_factory(server_base_url: str) -> RoundClient:
    return RoundClient(server_base_url=server_base_url)


def _default_federation_runtime_service_factory(
    server_base_url: str,
) -> FederationRuntimeService:
    return FederationRuntimeService(
        round_client=_default_round_client_factory(server_base_url)
    )


def create_app(
    *,
    pipeline_service: InferencePipelineService | None = None,
    scored_event_repository: ScoredEventRepository | None = None,
    prototype_runtime_service: PrototypeRuntimeService | None = None,
    scoring_service: ScoringService | None = None,
    round_client_factory: RoundClientFactory | None = None,
    federation_runtime_service_factory: FederationRuntimeServiceFactory | None = None,
) -> FastAPI:
    """Agent API 앱을 생성하고 override 가능한 기본 의존성을 연결한다."""
    app = FastAPI(title="TraceMind Agent", version="0.1.0")

    app.state.scored_event_repository = (
        scored_event_repository or ScoredEventRepository()
    )
    app.state.prototype_runtime_service = (
        prototype_runtime_service or PrototypeRuntimeService()
    )
    app.state.scoring_service = scoring_service or ScoringService()
    app.state.round_client_factory = (
        round_client_factory or _default_round_client_factory
    )
    app.state.federation_runtime_service_factory = (
        federation_runtime_service_factory
        or _default_federation_runtime_service_factory
    )
    if pipeline_service is not None:
        app.state.pipeline_service = pipeline_service

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(sync_router)
    app.include_router(training_router)
    return app


app = create_app()
