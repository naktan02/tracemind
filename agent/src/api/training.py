"""학습 라우터."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.infrastructure.repositories.captured_text.repository import (
    CapturedTextRepository,
)
from agent.src.infrastructure.repositories.training_usage_ledger_repository import (
    TrainingUsageLedgerRepository,
)
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.training_runtime.current_task.agent_training_task_runner_service import (  # noqa: E501
    AgentTrainingTaskRunnerService,
    AgentTrainingTaskRunRequest,
)
from agent.src.services.training_runtime.current_task.query_ssl_training_task_service import (  # noqa: E501
    AgentQuerySslTrainingTaskService,
)
from shared.src.contracts.training_contracts import TrainingTaskPayload


class RunCurrentTaskRequest(BaseModel):
    """run-current-task API 요청."""

    model_config = ConfigDict(extra="forbid")

    server_base_url: str = Field(
        validation_alias=AliasChoices("server_base_url", "serverBaseUrl"),
        description="Main server base URL.",
    )
    analysis_event_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="학습에 사용할 analysis event 보관 기간 (일). 기본 7일.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Pseudonymous agent UUID. 없으면 서버에 익명으로 전송.",
    )


class RunCurrentTaskResponse(BaseModel):
    """run-current-task API 응답."""

    model_config = ConfigDict(extra="forbid")

    status: str
    round_id: str | None = None
    task_id: str | None = None
    update_id: str | None = None
    example_count: int = 0
    accepted_count: int = 0
    message: str = ""


class TrainingStatusResponse(BaseModel):
    """현재 active task 상태 응답."""

    model_config = ConfigDict(extra="forbid")

    has_active_task: bool
    task: TrainingTaskPayload | None = None
    server_base_url: str | None = None


router = APIRouter(prefix="/api/v1/training", tags=["training"])

RoundClientFactory = Callable[[str], RoundClient]


# ------------------------------------------------------------------ #
# 의존성 providers                                                      #
# ------------------------------------------------------------------ #


def get_analysis_event_repository(request: Request) -> AnalysisEventRepository:
    """app.state에서 AnalysisEventRepository를 읽는다."""
    repo = getattr(request.app.state, "analysis_event_repository", None)
    if repo is None:
        raise RuntimeError(
            "AnalysisEventRepository가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.analysis_event_repository를 설정하세요."
        )
    return repo


def get_shared_adapter_runtime_service(request: Request) -> SharedAdapterRuntimeService:
    """app.state에서 SharedAdapterRuntimeService를 읽는다."""
    service = getattr(request.app.state, "shared_adapter_runtime_service", None)
    if service is None:
        raise RuntimeError(
            "SharedAdapterRuntimeService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.shared_adapter_runtime_service를 설정하세요."
        )
    return service


def get_shared_adapter_sync_service(request: Request) -> SharedAdapterSyncService:
    """app.state에서 SharedAdapterSyncService를 읽는다."""
    service = getattr(request.app.state, "shared_adapter_sync_service", None)
    if service is None:
        raise RuntimeError(
            "SharedAdapterSyncService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.shared_adapter_sync_service를 설정하세요."
        )
    return service


def get_round_client_factory(request: Request) -> RoundClientFactory:
    """app.state에서 RoundClient factory를 읽는다."""
    factory = getattr(request.app.state, "round_client_factory", None)
    if factory is None:
        raise RuntimeError(
            "RoundClient factory가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.round_client_factory를 설정하세요."
        )
    return factory


def get_training_usage_ledger_repository(
    request: Request,
) -> TrainingUsageLedgerRepository:
    """app.state에서 TrainingUsageLedgerRepository를 읽는다."""
    repository = getattr(request.app.state, "training_usage_ledger_repository", None)
    if repository is None:
        raise RuntimeError(
            "TrainingUsageLedgerRepository가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.training_usage_ledger_repository를 설정하세요."
        )
    return repository


AnalysisEventRepoDep = Annotated[
    AnalysisEventRepository,
    Depends(get_analysis_event_repository),
]
SharedAdapterRuntimeServiceDep = Annotated[
    SharedAdapterRuntimeService,
    Depends(get_shared_adapter_runtime_service),
]
SharedAdapterSyncServiceDep = Annotated[
    SharedAdapterSyncService,
    Depends(get_shared_adapter_sync_service),
]
RoundClientFactoryDep = Annotated[RoundClientFactory, Depends(get_round_client_factory)]
TrainingUsageLedgerRepoDep = Annotated[
    TrainingUsageLedgerRepository,
    Depends(get_training_usage_ledger_repository),
]


def get_training_task_runner_service(
    request: Request,
    repo: AnalysisEventRepoDep,
    shared_adapter_runtime_service: SharedAdapterRuntimeServiceDep,
    shared_adapter_sync_service: SharedAdapterSyncServiceDep,
    round_client_factory: RoundClientFactoryDep,
    training_usage_ledger_repository: TrainingUsageLedgerRepoDep,
) -> AgentTrainingTaskRunnerService:
    """run-current-task application service를 조립한다."""

    return AgentTrainingTaskRunnerService(
        analysis_event_repository=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        captured_text_repository=_get_optional_captured_text_repository(request),
        query_ssl_task_service=AgentQuerySslTrainingTaskService(
            usage_ledger_repository=training_usage_ledger_repository
        ),
    )


TrainingTaskRunnerServiceDep = Annotated[
    AgentTrainingTaskRunnerService,
    Depends(get_training_task_runner_service),
]


# ------------------------------------------------------------------ #
# 엔드포인트                                                            #
# ------------------------------------------------------------------ #


@router.post(
    "/run-current-task",
    response_model=RunCurrentTaskResponse,
    status_code=status.HTTP_200_OK,
)
def run_current_task(
    request: RunCurrentTaskRequest,
    runner_service: TrainingTaskRunnerServiceDep,
) -> RunCurrentTaskResponse:
    """현재 active task를 읽어 로컬 학습을 실행하고 update를 업로드한다.

    학습 예시가 없으면 INSUFFICIENT_EXAMPLES 상태를 반환한다.
    active round 자체가 없으면 NO_ACTIVE_TASK 상태를 반환한다.
    두 경우 모두 200 OK로 응답하고 status 필드로 구분한다.
    """
    try:
        result = runner_service.run_current_task(
            AgentTrainingTaskRunRequest(
                server_base_url=request.server_base_url,
                analysis_event_days=request.analysis_event_days,
                agent_id=request.agent_id,
            )
        )
        return RunCurrentTaskResponse(
            status=result.status,
            round_id=result.round_id,
            task_id=result.task_id,
            update_id=result.update_id,
            example_count=result.example_count,
            accepted_count=result.accepted_count,
            message=result.message,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"서버 통신 오류: {error}",
        ) from error


@router.get(
    "/status",
    response_model=TrainingStatusResponse,
    status_code=status.HTTP_200_OK,
)
def get_training_status(
    server_base_url: str,
    round_client_factory: RoundClientFactoryDep,
) -> TrainingStatusResponse:
    """현재 active round/task 조회."""
    try:
        client = round_client_factory(server_base_url)
        task_payload = client.fetch_current_task()
        return TrainingStatusResponse(
            has_active_task=task_payload is not None,
            task=task_payload,
            server_base_url=server_base_url,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"서버 통신 오류: {error}",
        ) from error


# ------------------------------------------------------------------ #
# 내부 헬퍼                                                             #
# ------------------------------------------------------------------ #


def _get_optional_captured_text_repository(
    request: Request,
) -> CapturedTextRepository | None:
    repository = getattr(request.app.state, "captured_text_repository", None)
    if repository is None:
        return None
    return repository
