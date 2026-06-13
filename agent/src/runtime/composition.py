"""Agent runtime object graph 조립."""

from __future__ import annotations

from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
)
from agent.src.infrastructure.repositories.child_support_repository import (
    ChildSupportConversationRepository,
)
from agent.src.infrastructure.repositories.family_access_repository import (
    FamilyAccessRepository,
)
from agent.src.infrastructure.repositories.training_usage_ledger_repository import (
    TrainingUsageLedgerRepository,
)
from agent.src.infrastructure.repositories.wellbeing_settings_repository import (
    WellbeingSettingsRepository,
)
from agent.src.infrastructure.repositories.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.runtime.state import AgentRuntimeState, RoundClientFactory
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.inference.pipeline_factory import build_default_pipeline_service
from agent.src.services.inference.pipeline_service import InferencePipelineService
from agent.src.services.ingest.captured_text_lifecycle_service import (
    CapturedTextLifecycleService,
    build_captured_text_lifecycle_service_from_env,
)
from agent.src.services.ingest.captured_text_view_generation_service import (
    CapturedTextViewGenerationService,
)
from agent.src.services.ingest.captured_text_view_provider_factory import (
    build_captured_text_view_generation_service_from_env,
)
from agent.src.services.wellbeing.child_support.context_provider import (
    ChildSupportContextProvider,
)
from agent.src.services.wellbeing.child_support.llm_provider import (
    ChildSupportLlmProvider,
    build_child_support_llm_provider_from_env,
)
from agent.src.services.wellbeing.child_support.service import (
    ChildSupportCoachService,
)
from agent.src.services.wellbeing.family_access.parent_auth_adapter import (
    ParentAuthService,
)
from agent.src.services.wellbeing.family_access.service import FamilyAccessService
from agent.src.services.wellbeing.signal.projection_service import (
    WellbeingSignalProjectionService,
)
from agent.src.services.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.services.wellbeing.signal.timeseries_service import (
    WellbeingTimeseriesService,
)
from agent.src.services.wellbeing.space_web.projection_service import (
    WellbeingSpaceWebProjectionService,
)


def default_round_client_factory(server_base_url: str) -> RoundClient:
    """기본 RoundClient Adapter를 만든다."""

    return RoundClient(server_base_url=server_base_url)


def build_agent_runtime_state(
    *,
    pipeline_service: InferencePipelineService | None = None,
    analysis_event_repository: AnalysisEventRepository | None = None,
    captured_text_repository: CapturedTextRepository | None = None,
    training_usage_ledger_repository: TrainingUsageLedgerRepository | None = None,
    captured_text_view_generation_service: (
        CapturedTextViewGenerationService | None
    ) = None,
    captured_text_lifecycle_service: CapturedTextLifecycleService | None = None,
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
    auto_configure_pipeline: bool = False,
) -> AgentRuntimeState:
    """Agent process 기본 runtime object graph를 만든다."""

    analysis_repo = analysis_event_repository or AnalysisEventRepository()
    captured_repo = captured_text_repository or CapturedTextRepository()
    usage_ledger_repo = (
        training_usage_ledger_repository or TrainingUsageLedgerRepository()
    )
    lifecycle_service = (
        captured_text_lifecycle_service
        or build_captured_text_lifecycle_service_from_env()
    )
    view_generation_service = (
        captured_text_view_generation_service
        or build_captured_text_view_generation_service_from_env(
            repository=captured_repo,
        )
    )
    child_support_repo = (
        child_support_conversation_repository or ChildSupportConversationRepository()
    )
    wellbeing_snapshot_repo = (
        wellbeing_snapshot_repository or WellbeingSnapshotRepository()
    )
    family_access_repo = family_access_repository or FamilyAccessRepository()
    wellbeing_settings_repo = (
        wellbeing_settings_repository or WellbeingSettingsRepository()
    )
    shared_runtime_service = (
        shared_adapter_runtime_service or SharedAdapterRuntimeService()
    )
    shared_sync_service = shared_adapter_sync_service or SharedAdapterSyncService()

    wellbeing_projection_service = WellbeingSignalProjectionService(
        analysis_event_repository=analysis_repo,
        snapshot_repository=wellbeing_snapshot_repo,
    )
    wellbeing_summary_service = WellbeingSummaryService(
        repository=wellbeing_snapshot_repo,
        projection_service=wellbeing_projection_service,
    )
    wellbeing_timeseries_service = WellbeingTimeseriesService(
        repository=wellbeing_snapshot_repo,
        projection_service=wellbeing_projection_service,
    )
    wellbeing_space_web_service = WellbeingSpaceWebProjectionService(
        analysis_event_repository=analysis_repo,
    )
    family_access_service = FamilyAccessService(
        repository=family_access_repo,
        settings_repository=wellbeing_settings_repo,
    )
    parent_auth_service = ParentAuthService(
        family_access_service=family_access_service,
    )
    resolved_child_support_coach_service = (
        child_support_coach_service
        or ChildSupportCoachService(
            conversation_repository=child_support_repo,
            context_provider=ChildSupportContextProvider(
                summary_service=wellbeing_summary_service,
                conversation_repository=child_support_repo,
            ),
            llm_provider=(
                child_support_llm_provider
                or build_child_support_llm_provider_from_env()
            ),
        )
    )
    resolved_pipeline_service = _resolve_pipeline_service(
        pipeline_service=pipeline_service,
        auto_configure_pipeline=auto_configure_pipeline,
        analysis_event_repository=analysis_repo,
        shared_adapter_runtime_service=shared_runtime_service,
        captured_text_view_generation_service=view_generation_service,
    )

    return AgentRuntimeState(
        analysis_event_repository=analysis_repo,
        captured_text_repository=captured_repo,
        training_usage_ledger_repository=usage_ledger_repo,
        captured_text_lifecycle_service=lifecycle_service,
        captured_text_view_generation_service=view_generation_service,
        child_support_conversation_repository=child_support_repo,
        wellbeing_snapshot_repository=wellbeing_snapshot_repo,
        family_access_repository=family_access_repo,
        wellbeing_settings_repository=wellbeing_settings_repo,
        shared_adapter_runtime_service=shared_runtime_service,
        shared_adapter_sync_service=shared_sync_service,
        wellbeing_projection_service=wellbeing_projection_service,
        wellbeing_summary_service=wellbeing_summary_service,
        wellbeing_timeseries_service=wellbeing_timeseries_service,
        wellbeing_space_web_service=wellbeing_space_web_service,
        family_access_service=family_access_service,
        parent_auth_service=parent_auth_service,
        child_support_coach_service=resolved_child_support_coach_service,
        round_client_factory=round_client_factory or default_round_client_factory,
        pipeline_service=resolved_pipeline_service,
    )


def _resolve_pipeline_service(
    *,
    pipeline_service: InferencePipelineService | None,
    auto_configure_pipeline: bool,
    analysis_event_repository: AnalysisEventRepository,
    shared_adapter_runtime_service: SharedAdapterRuntimeService,
    captured_text_view_generation_service: CapturedTextViewGenerationService,
) -> InferencePipelineService | None:
    if pipeline_service is not None:
        return pipeline_service
    if not auto_configure_pipeline:
        return None
    return build_default_pipeline_service(
        analysis_event_repository=analysis_event_repository,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        translation_service=(
            captured_text_view_generation_service.translation_provider
        ),
    )
