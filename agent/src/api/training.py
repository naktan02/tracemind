"""학습 라우터."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
)
from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.services.assets.prototypes.runtime_service import PrototypeRuntimeService
from agent.src.services.assets.prototypes.sync_service import PrototypeSyncService
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.federation.rounds.runtime_service import (
    FederationRuntimeService,
)
from agent.src.services.training.execution.agent_training_task_runner_service import (
    AgentTrainingTaskRunnerService,
    AgentTrainingTaskRunRequest,
)
from shared.src.contracts.training_contracts import TrainingTaskPayload
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


class RunCurrentTaskRequest(BaseModel):
    """run-current-task API 요청."""

    model_config = ConfigDict(extra="forbid")

    server_base_url: str
    scored_event_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="학습에 사용할 scored event 보관 기간 (일). 기본 7일.",
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
FederationRuntimeServiceFactory = Callable[[str], FederationRuntimeService]


# ------------------------------------------------------------------ #
# 의존성 providers                                                      #
# ------------------------------------------------------------------ #


def get_scored_event_repository(request: Request) -> ScoredEventRepository:
    """app.state에서 ScoredEventRepository를 읽는다."""
    repo = getattr(request.app.state, "scored_event_repository", None)
    if repo is None:
        raise RuntimeError(
            "ScoredEventRepository가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.scored_event_repository를 설정하세요."
        )
    return repo


def get_prototype_runtime_service(request: Request) -> PrototypeRuntimeService:
    """app.state에서 PrototypeRuntimeService를 읽는다."""
    service = getattr(request.app.state, "prototype_runtime_service", None)
    if service is None:
        raise RuntimeError(
            "PrototypeRuntimeService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.prototype_runtime_service를 설정하세요."
        )
    return service


def get_prototype_sync_service(request: Request) -> PrototypeSyncService:
    """app.state에서 PrototypeSyncService를 읽는다."""
    service = getattr(request.app.state, "prototype_sync_service", None)
    if service is None:
        raise RuntimeError(
            "PrototypeSyncService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.prototype_sync_service를 설정하세요."
        )
    return service


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


def get_federation_runtime_service_factory(
    request: Request,
) -> FederationRuntimeServiceFactory:
    """app.state에서 FederationRuntimeService factory를 읽는다."""
    factory = getattr(request.app.state, "federation_runtime_service_factory", None)
    if factory is None:
        raise RuntimeError(
            "FederationRuntimeService factory가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.federation_runtime_service_factory를 설정하세요."
        )
    return factory


ScoredEventRepoDep = Annotated[
    ScoredEventRepository,
    Depends(get_scored_event_repository),
]
ProtoServiceDep = Annotated[
    PrototypeRuntimeService,
    Depends(get_prototype_runtime_service),
]
PrototypeSyncServiceDep = Annotated[
    PrototypeSyncService,
    Depends(get_prototype_sync_service),
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
FederationRuntimeFactoryDep = Annotated[
    FederationRuntimeServiceFactory,
    Depends(get_federation_runtime_service_factory),
]


def get_training_task_runner_service(
    request: Request,
    repo: ScoredEventRepoDep,
    proto_service: ProtoServiceDep,
    proto_sync_service: PrototypeSyncServiceDep,
    shared_adapter_runtime_service: SharedAdapterRuntimeServiceDep,
    shared_adapter_sync_service: SharedAdapterSyncServiceDep,
    round_client_factory: RoundClientFactoryDep,
    runtime_factory: FederationRuntimeFactoryDep,
) -> AgentTrainingTaskRunnerService:
    """run-current-task application service를 조립한다."""

    return AgentTrainingTaskRunnerService(
        scored_event_repository=repo,
        prototype_runtime_service=proto_service,
        prototype_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        federation_runtime_service_factory=runtime_factory,
        captured_text_repository=_get_optional_captured_text_repository(request),
        embedding_adapter=_get_optional_training_embedding_adapter(request),
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
                scored_event_days=request.scored_event_days,
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


def _get_optional_training_embedding_adapter(
    request: Request,
) -> EmbeddingAdapter | None:
    adapter = getattr(request.app.state, "training_embedding_adapter", None)
    if adapter is not None:
        return adapter
    pipeline_service = getattr(request.app.state, "pipeline_service", None)
    embedding_service = getattr(pipeline_service, "embedding_service", None)
    pipeline_adapter = getattr(embedding_service, "adapter", None)
    return pipeline_adapter
