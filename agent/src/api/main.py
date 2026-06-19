"""Agent API 진입점."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.src.api.captured_text import router as captured_text_router
from agent.src.api.child_support import router as child_support_router
from agent.src.api.family_access import router as family_access_router
from agent.src.api.health import router as health_router
from agent.src.api.ingest import router as ingest_router
from agent.src.api.runtime_profile import router as runtime_profile_router
from agent.src.api.sync import router as sync_router
from agent.src.api.training import router as training_router
from agent.src.api.typing_segments import router as typing_segments_router
from agent.src.api.wellbeing import router as wellbeing_router
from agent.src.config.env_file import load_agent_env_files
from agent.src.runtime import env as runtime_env
from agent.src.runtime.composition import (
    build_agent_runtime_state,
)
from agent.src.runtime.state import RoundClientFactory, install_agent_runtime_state

if TYPE_CHECKING:
    from agent.src.features.assets.shared_adapters.runtime_service import (
        SharedAdapterRuntimeService,
    )
    from agent.src.features.assets.shared_adapters.sync_service import (
        SharedAdapterSyncService,
    )
    from agent.src.features.captured_text.lifecycle import (
        CapturedTextLifecycleService,
    )
    from agent.src.features.captured_text.storage.repository import (
        CapturedTextRepository,
    )
    from agent.src.features.captured_text.view_generation.service import (
        CapturedTextViewGenerationService,
    )
    from agent.src.features.inference.pipeline_service import InferencePipelineService
    from agent.src.features.runtime_profile.repository import RuntimeProfileRepository
    from agent.src.features.runtime_profile.sync_service import (
        RuntimeProfileSyncService,
    )
    from agent.src.features.training_runtime.storage.training_usage_ledger_repository import (  # noqa: E501
        TrainingUsageLedgerRepository,
    )
    from agent.src.features.wellbeing.child_support.llm_provider import (
        ChildSupportLlmProvider,
    )
    from agent.src.features.wellbeing.child_support.service import (
        ChildSupportCoachService,
    )
    from agent.src.features.wellbeing.storage.child_support_repository import (
        ChildSupportConversationRepository,
    )
    from agent.src.features.wellbeing.storage.family_access_repository import (
        FamilyAccessRepository,
    )
    from agent.src.features.wellbeing.storage.wellbeing_settings_repository import (
        WellbeingSettingsRepository,
    )
    from agent.src.features.wellbeing.storage.wellbeing_snapshot_repository import (
        WellbeingSnapshotRepository,
    )
    from agent.src.infrastructure.repositories.analysis_event_repository import (
        AnalysisEventRepository,
    )

load_agent_env_files()

FAMILY_EXTENSION_ALLOWED_ORIGINS_ENV = runtime_env.FAMILY_EXTENSION_ALLOWED_ORIGINS_ENV
DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS = (
    runtime_env.DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS
)
load_family_extension_allowed_origins_from_env = (
    runtime_env.load_family_extension_allowed_origins_from_env
)


def create_app(
    *,
    pipeline_service: InferencePipelineService | None = None,
    analysis_event_repository: AnalysisEventRepository | None = None,
    captured_text_repository: CapturedTextRepository | None = None,
    training_usage_ledger_repository: TrainingUsageLedgerRepository | None = None,
    captured_text_view_generation_service: (
        CapturedTextViewGenerationService | None
    ) = None,
    captured_text_lifecycle_service: CapturedTextLifecycleService | None = None,
    runtime_profile_repository: RuntimeProfileRepository | None = None,
    runtime_profile_sync_service: RuntimeProfileSyncService | None = None,
    child_support_conversation_repository: (
        ChildSupportConversationRepository | None
    ) = None,
    wellbeing_snapshot_repository: WellbeingSnapshotRepository | None = None,
    family_access_repository: FamilyAccessRepository | None = None,
    wellbeing_settings_repository: WellbeingSettingsRepository | None = None,
    shared_adapter_runtime_service: SharedAdapterRuntimeService | None = None,
    shared_adapter_sync_service: SharedAdapterSyncService | None = None,
    child_support_coach_service: ChildSupportCoachService | None = None,
    child_support_llm_provider: ChildSupportLlmProvider | None = None,
    round_client_factory: RoundClientFactory | None = None,
    family_extension_allowed_origins: tuple[str, ...] | None = None,
    auto_configure_pipeline: bool = False,
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

    runtime_state = build_agent_runtime_state(
        pipeline_service=pipeline_service,
        analysis_event_repository=analysis_event_repository,
        captured_text_repository=captured_text_repository,
        training_usage_ledger_repository=training_usage_ledger_repository,
        captured_text_view_generation_service=captured_text_view_generation_service,
        captured_text_lifecycle_service=captured_text_lifecycle_service,
        runtime_profile_repository=runtime_profile_repository,
        runtime_profile_sync_service=runtime_profile_sync_service,
        child_support_conversation_repository=child_support_conversation_repository,
        wellbeing_snapshot_repository=wellbeing_snapshot_repository,
        family_access_repository=family_access_repository,
        wellbeing_settings_repository=wellbeing_settings_repository,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        child_support_coach_service=child_support_coach_service,
        child_support_llm_provider=child_support_llm_provider,
        round_client_factory=round_client_factory,
        auto_configure_pipeline=auto_configure_pipeline,
    )
    install_agent_runtime_state(app.state, runtime_state)

    app.include_router(health_router)
    app.include_router(captured_text_router)
    app.include_router(child_support_router)
    app.include_router(family_access_router)
    app.include_router(ingest_router)
    app.include_router(runtime_profile_router)
    app.include_router(sync_router)
    app.include_router(training_router)
    app.include_router(typing_segments_router)
    app.include_router(wellbeing_router)
    return app


try:
    app = create_app(auto_configure_pipeline=True)
except ValueError as exc:
    message = str(exc)
    if (
        "TRACEMIND_AGENT_SCORING_BACKEND" not in message
        and "server_base_url is required to materialize PEFT" not in message
    ):
        raise
    app = create_app(auto_configure_pipeline=False)
