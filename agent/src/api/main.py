"""Agent API 진입점."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from agent.src.api.health import router as health_router
from agent.src.api.ingest import router as ingest_router
from agent.src.api.sync import router as sync_router
from agent.src.api.training import router as training_router
from agent.src.api.wellbeing import router as wellbeing_router
from agent.src.infrastructure.repositories.parent_auth_repository import (
    ParentAuthRepository,
)
from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRepository,
)
from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.infrastructure.repositories.wellbeing_settings_repository import (
    WellbeingSettingsRepository,
)
from agent.src.infrastructure.repositories.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.services.assets.prototypes.runtime_service import PrototypeRuntimeService
from agent.src.services.assets.prototypes.sync_service import PrototypeSyncService
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.federation.rounds.runtime_service import (
    FederationRuntimeService,
)
from agent.src.services.inference.pipeline_service import InferencePipelineService
from agent.src.services.wellbeing.auth_service import ParentAuthService
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from agent.src.services.wellbeing.timeseries_service import (
    WellbeingTimeseriesService,
)

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
    query_buffer_repository: QueryBufferRepository | None = None,
    wellbeing_snapshot_repository: WellbeingSnapshotRepository | None = None,
    parent_auth_repository: ParentAuthRepository | None = None,
    wellbeing_settings_repository: WellbeingSettingsRepository | None = None,
    prototype_runtime_service: PrototypeRuntimeService | None = None,
    prototype_sync_service: PrototypeSyncService | None = None,
    round_client_factory: RoundClientFactory | None = None,
    federation_runtime_service_factory: FederationRuntimeServiceFactory | None = None,
) -> FastAPI:
    """Agent API 앱을 생성하고 override 가능한 기본 의존성을 연결한다."""
    app = FastAPI(title="TraceMind Agent", version="0.1.0")

    app.state.scored_event_repository = (
        scored_event_repository or ScoredEventRepository()
    )
    app.state.query_buffer_repository = (
        query_buffer_repository or QueryBufferRepository()
    )
    app.state.wellbeing_snapshot_repository = (
        wellbeing_snapshot_repository or WellbeingSnapshotRepository()
    )
    app.state.parent_auth_repository = (
        parent_auth_repository or ParentAuthRepository()
    )
    app.state.wellbeing_settings_repository = (
        wellbeing_settings_repository or WellbeingSettingsRepository()
    )
    app.state.prototype_runtime_service = (
        prototype_runtime_service or PrototypeRuntimeService()
    )
    app.state.prototype_sync_service = prototype_sync_service or PrototypeSyncService()
    app.state.wellbeing_summary_service = WellbeingSummaryService(
        repository=app.state.wellbeing_snapshot_repository
    )
    app.state.wellbeing_timeseries_service = WellbeingTimeseriesService(
        repository=app.state.wellbeing_snapshot_repository
    )
    app.state.parent_auth_service = ParentAuthService(
        repository=app.state.parent_auth_repository,
        settings_repository=app.state.wellbeing_settings_repository,
    )
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
    app.include_router(wellbeing_router)
    return app


app = create_app()
