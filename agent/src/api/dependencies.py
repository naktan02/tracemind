"""FastAPI dependency glue for agent runtime state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypeVar, cast

from fastapi import Depends, HTTPException, Request, status

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
from agent.src.features.runtime_profile.sync_service import RuntimeProfileSyncService
from agent.src.features.training_runtime.storage.training_usage_ledger_repository import (  # noqa: E501
    TrainingUsageLedgerRepository,
)
from agent.src.features.wellbeing.child_support.service import (
    ChildSupportCoachService,
)
from agent.src.features.wellbeing.family_access.parent_auth_adapter import (
    ParentAuthService,
)
from agent.src.features.wellbeing.family_access.service import FamilyAccessService
from agent.src.features.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.features.wellbeing.signal.timeseries_service import (
    WellbeingTimeseriesService,
)
from agent.src.features.wellbeing.space_web.projection_service import (
    WellbeingSpaceWebProjectionService,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.runtime.state import RoundClientFactory

RuntimeStateT = TypeVar("RuntimeStateT")


def get_required_app_state(
    request: Request,
    name: str,
    *,
    display_name: str | None = None,
) -> RuntimeStateT:
    """FastAPI app.state에서 필수 runtime 객체를 읽는다."""

    value = getattr(request.app.state, name, None)
    if value is None:
        label = display_name or name
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"{label}가 app.state에 설정되지 않았습니다. "
                f"앱 생성 시 app.state.{name}을 설정하세요."
            ),
        )
    return cast(RuntimeStateT, value)


def get_optional_app_state(
    request: Request,
    name: str,
) -> RuntimeStateT | None:
    """FastAPI app.state에서 선택 runtime 객체를 읽는다."""

    return cast(RuntimeStateT | None, getattr(request.app.state, name, None))


def get_or_create_app_state(
    request: Request,
    name: str,
    factory: Callable[[], RuntimeStateT],
) -> RuntimeStateT:
    """FastAPI app.state에 없으면 factory 결과를 설치하고 반환한다."""

    value = getattr(request.app.state, name, None)
    if value is None:
        value = factory()
        setattr(request.app.state, name, value)
    return cast(RuntimeStateT, value)


def get_pipeline_service(request: Request) -> InferencePipelineService:
    return get_required_app_state(
        request,
        "pipeline_service",
        display_name="InferencePipelineService",
    )


def get_optional_pipeline_service(
    request: Request,
) -> InferencePipelineService | None:
    return get_optional_app_state(request, "pipeline_service")


def get_analysis_event_repository(request: Request) -> AnalysisEventRepository:
    return get_required_app_state(
        request,
        "analysis_event_repository",
        display_name="AnalysisEventRepository",
    )


def get_captured_text_repository(request: Request) -> CapturedTextRepository:
    return get_required_app_state(
        request,
        "captured_text_repository",
        display_name="CapturedTextRepository",
    )


def get_captured_text_lifecycle_service(
    request: Request,
) -> CapturedTextLifecycleService:
    return get_required_app_state(
        request,
        "captured_text_lifecycle_service",
        display_name="CapturedTextLifecycleService",
    )


def get_captured_text_view_generation_service(
    request: Request,
) -> CapturedTextViewGenerationService:
    return get_required_app_state(
        request,
        "captured_text_view_generation_service",
        display_name="CapturedTextViewGenerationService",
    )


def get_shared_adapter_runtime_service(request: Request) -> SharedAdapterRuntimeService:
    return get_required_app_state(
        request,
        "shared_adapter_runtime_service",
        display_name="SharedAdapterRuntimeService",
    )


def get_runtime_profile_repository(request: Request) -> RuntimeProfileRepository:
    return get_required_app_state(
        request,
        "runtime_profile_repository",
        display_name="RuntimeProfileRepository",
    )


def get_runtime_profile_sync_service(request: Request) -> RuntimeProfileSyncService:
    return get_required_app_state(
        request,
        "runtime_profile_sync_service",
        display_name="RuntimeProfileSyncService",
    )


def get_shared_adapter_sync_service(request: Request) -> SharedAdapterSyncService:
    return get_required_app_state(
        request,
        "shared_adapter_sync_service",
        display_name="SharedAdapterSyncService",
    )


def get_round_client_factory(request: Request) -> RoundClientFactory:
    return get_required_app_state(
        request,
        "round_client_factory",
        display_name="RoundClient factory",
    )


def get_training_usage_ledger_repository(
    request: Request,
) -> TrainingUsageLedgerRepository:
    return get_required_app_state(
        request,
        "training_usage_ledger_repository",
        display_name="TrainingUsageLedgerRepository",
    )


def get_wellbeing_summary_service(request: Request) -> WellbeingSummaryService:
    return get_required_app_state(
        request,
        "wellbeing_summary_service",
        display_name="WellbeingSummaryService",
    )


def get_wellbeing_timeseries_service(
    request: Request,
) -> WellbeingTimeseriesService:
    return get_required_app_state(
        request,
        "wellbeing_timeseries_service",
        display_name="WellbeingTimeseriesService",
    )


def get_wellbeing_space_web_service(
    request: Request,
) -> WellbeingSpaceWebProjectionService:
    return get_required_app_state(
        request,
        "wellbeing_space_web_service",
        display_name="WellbeingSpaceWebProjectionService",
    )


def get_parent_auth_service(request: Request) -> ParentAuthService:
    return get_required_app_state(
        request,
        "parent_auth_service",
        display_name="ParentAuthService",
    )


def get_child_support_coach_service(request: Request) -> ChildSupportCoachService:
    return get_required_app_state(
        request,
        "child_support_coach_service",
        display_name="ChildSupportCoachService",
    )


def get_family_access_service(request: Request) -> FamilyAccessService:
    return get_required_app_state(
        request,
        "family_access_service",
        display_name="FamilyAccessService",
    )


PipelineServiceDep = Annotated[
    InferencePipelineService,
    Depends(get_pipeline_service),
]
OptionalPipelineServiceDep = Annotated[
    InferencePipelineService | None,
    Depends(get_optional_pipeline_service),
]
AnalysisEventRepositoryDep = Annotated[
    AnalysisEventRepository,
    Depends(get_analysis_event_repository),
]
CapturedTextRepositoryDep = Annotated[
    CapturedTextRepository,
    Depends(get_captured_text_repository),
]
CapturedTextLifecycleServiceDep = Annotated[
    CapturedTextLifecycleService,
    Depends(get_captured_text_lifecycle_service),
]
CapturedTextViewGenerationServiceDep = Annotated[
    CapturedTextViewGenerationService,
    Depends(get_captured_text_view_generation_service),
]
SharedAdapterRuntimeServiceDep = Annotated[
    SharedAdapterRuntimeService,
    Depends(get_shared_adapter_runtime_service),
]
RuntimeProfileRepositoryDep = Annotated[
    RuntimeProfileRepository,
    Depends(get_runtime_profile_repository),
]
RuntimeProfileSyncServiceDep = Annotated[
    RuntimeProfileSyncService,
    Depends(get_runtime_profile_sync_service),
]
SharedAdapterSyncServiceDep = Annotated[
    SharedAdapterSyncService,
    Depends(get_shared_adapter_sync_service),
]
RoundClientFactoryDep = Annotated[
    RoundClientFactory,
    Depends(get_round_client_factory),
]
TrainingUsageLedgerRepositoryDep = Annotated[
    TrainingUsageLedgerRepository,
    Depends(get_training_usage_ledger_repository),
]
WellbeingSummaryServiceDep = Annotated[
    WellbeingSummaryService,
    Depends(get_wellbeing_summary_service),
]
WellbeingTimeseriesServiceDep = Annotated[
    WellbeingTimeseriesService,
    Depends(get_wellbeing_timeseries_service),
]
WellbeingSpaceWebServiceDep = Annotated[
    WellbeingSpaceWebProjectionService,
    Depends(get_wellbeing_space_web_service),
]
ParentAuthServiceDep = Annotated[
    ParentAuthService,
    Depends(get_parent_auth_service),
]
ChildSupportCoachServiceDep = Annotated[
    ChildSupportCoachService,
    Depends(get_child_support_coach_service),
]
FamilyAccessServiceDep = Annotated[
    FamilyAccessService,
    Depends(get_family_access_service),
]
