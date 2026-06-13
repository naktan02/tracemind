"""Agent runtime state 설치 규칙."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

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
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from agent.src.services.captured_text.lifecycle import (
    CapturedTextLifecycleService,
)
from agent.src.services.captured_text.view_generation.service import (
    CapturedTextViewGenerationService,
)
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.inference.pipeline_service import InferencePipelineService
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

RoundClientFactory = Callable[[str], RoundClient]


@dataclass(slots=True)
class AgentRuntimeState:
    """FastAPI app.state에 설치할 agent runtime object 묶음."""

    analysis_event_repository: AnalysisEventRepository
    captured_text_repository: CapturedTextRepository
    training_usage_ledger_repository: TrainingUsageLedgerRepository
    captured_text_lifecycle_service: CapturedTextLifecycleService
    captured_text_view_generation_service: CapturedTextViewGenerationService
    child_support_conversation_repository: ChildSupportConversationRepository
    wellbeing_snapshot_repository: WellbeingSnapshotRepository
    family_access_repository: FamilyAccessRepository
    wellbeing_settings_repository: WellbeingSettingsRepository
    shared_adapter_runtime_service: SharedAdapterRuntimeService
    shared_adapter_sync_service: SharedAdapterSyncService
    wellbeing_projection_service: WellbeingSignalProjectionService
    wellbeing_summary_service: WellbeingSummaryService
    wellbeing_timeseries_service: WellbeingTimeseriesService
    wellbeing_space_web_service: WellbeingSpaceWebProjectionService
    family_access_service: FamilyAccessService
    parent_auth_service: ParentAuthService
    child_support_coach_service: ChildSupportCoachService
    round_client_factory: RoundClientFactory
    pipeline_service: InferencePipelineService | None = None


def install_agent_runtime_state(
    app_state: object,
    runtime_state: AgentRuntimeState,
) -> None:
    """AgentRuntimeState 필드를 FastAPI app.state에 설치한다."""

    for field in fields(runtime_state):
        value = getattr(runtime_state, field.name)
        if value is None:
            continue
        setattr(app_state, field.name, value)
