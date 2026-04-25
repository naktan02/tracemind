"""Agent API 진입점."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.src.api.child_support import router as child_support_router
from agent.src.api.family_access import router as family_access_router
from agent.src.api.health import router as health_router
from agent.src.api.ingest import router as ingest_router
from agent.src.api.sync import router as sync_router
from agent.src.api.training import router as training_router
from agent.src.api.wellbeing import router as wellbeing_router
from agent.src.infrastructure.repositories.child_support_repository import (
    ChildSupportConversationRepository,
)
from agent.src.infrastructure.repositories.family_access_repository import (
    FamilyAccessRepository,
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
from agent.src.services.wellbeing.child_support_context_provider import (
    ChildSupportContextProvider,
)
from agent.src.services.wellbeing.child_support_llm_provider import (
    ChildSupportLlmProvider,
    build_child_support_llm_provider_from_env,
)
from agent.src.services.wellbeing.child_support_service import (
    ChildSupportCoachService,
)
from agent.src.services.wellbeing.family_access_service import FamilyAccessService
from agent.src.services.wellbeing.projection_service import (
    WellbeingSignalProjectionService,
)
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from agent.src.services.wellbeing.timeseries_service import (
    WellbeingTimeseriesService,
)

RoundClientFactory = Callable[[str], RoundClient]
FederationRuntimeServiceFactory = Callable[[str], FederationRuntimeService]
FAMILY_EXTENSION_ALLOWED_ORIGINS_ENV = "FAMILY_EXTENSION_ALLOWED_ORIGINS"
DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS = (
    "http://localhost:5174",
    "http://127.0.0.1:5174",
)


def load_family_extension_allowed_origins_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """family_extension dev server가 접근할 수 있는 origin 목록을 읽는다."""

    effective_environ = os.environ if environ is None else environ
    raw_value = effective_environ.get(FAMILY_EXTENSION_ALLOWED_ORIGINS_ENV, "")
    origins = tuple(origin.strip() for origin in raw_value.split(",") if origin.strip())
    return origins or DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS


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
    child_support_conversation_repository: (
        ChildSupportConversationRepository | None
    ) = None,
    wellbeing_snapshot_repository: WellbeingSnapshotRepository | None = None,
    family_access_repository: FamilyAccessRepository | None = None,
    wellbeing_settings_repository: WellbeingSettingsRepository | None = None,
    prototype_runtime_service: PrototypeRuntimeService | None = None,
    prototype_sync_service: PrototypeSyncService | None = None,
    child_support_coach_service: ChildSupportCoachService | None = None,
    child_support_llm_provider: ChildSupportLlmProvider | None = None,
    round_client_factory: RoundClientFactory | None = None,
    federation_runtime_service_factory: FederationRuntimeServiceFactory | None = None,
    family_extension_allowed_origins: tuple[str, ...] | None = None,
) -> FastAPI:
    """Agent API 앱을 생성하고 override 가능한 기본 의존성을 연결한다."""
    app = FastAPI(title="TraceMind Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            family_extension_allowed_origins
            or load_family_extension_allowed_origins_from_env()
        ),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.scored_event_repository = (
        scored_event_repository or ScoredEventRepository()
    )
    app.state.query_buffer_repository = (
        query_buffer_repository or QueryBufferRepository()
    )
    app.state.child_support_conversation_repository = (
        child_support_conversation_repository or ChildSupportConversationRepository()
    )
    app.state.wellbeing_snapshot_repository = (
        wellbeing_snapshot_repository or WellbeingSnapshotRepository()
    )
    app.state.family_access_repository = (
        family_access_repository or FamilyAccessRepository()
    )
    app.state.wellbeing_settings_repository = (
        wellbeing_settings_repository or WellbeingSettingsRepository()
    )
    app.state.prototype_runtime_service = (
        prototype_runtime_service or PrototypeRuntimeService()
    )
    app.state.prototype_sync_service = prototype_sync_service or PrototypeSyncService()
    app.state.wellbeing_projection_service = WellbeingSignalProjectionService(
        scored_event_repository=app.state.scored_event_repository,
        snapshot_repository=app.state.wellbeing_snapshot_repository,
    )
    app.state.wellbeing_summary_service = WellbeingSummaryService(
        repository=app.state.wellbeing_snapshot_repository,
        projection_service=app.state.wellbeing_projection_service,
    )
    app.state.wellbeing_timeseries_service = WellbeingTimeseriesService(
        repository=app.state.wellbeing_snapshot_repository,
        projection_service=app.state.wellbeing_projection_service,
    )
    app.state.family_access_service = FamilyAccessService(
        repository=app.state.family_access_repository,
        settings_repository=app.state.wellbeing_settings_repository,
    )
    app.state.parent_auth_service = ParentAuthService(
        family_access_service=app.state.family_access_service,
    )
    app.state.child_support_coach_service = (
        child_support_coach_service
        or ChildSupportCoachService(
            conversation_repository=(app.state.child_support_conversation_repository),
            context_provider=ChildSupportContextProvider(
                summary_service=app.state.wellbeing_summary_service,
                query_buffer_repository=app.state.query_buffer_repository,
                conversation_repository=(
                    app.state.child_support_conversation_repository
                ),
            ),
            llm_provider=(
                child_support_llm_provider
                or build_child_support_llm_provider_from_env()
            ),
        )
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
    app.include_router(child_support_router)
    app.include_router(family_access_router)
    app.include_router(ingest_router)
    app.include_router(sync_router)
    app.include_router(training_router)
    app.include_router(wellbeing_router)
    return app


app = create_app()
