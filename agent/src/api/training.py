"""학습 라우터."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.services.federation import StoredEventTrainingExampleBuildRequest
from agent.src.services.federation.round_client import RoundClient
from agent.src.services.federation.runtime_service import (
    FederationRunResult,
    FederationRuntimeService,
)
from agent.src.services.federation.training_example_service import (
    TrainingExampleService,
)
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.prototypes.runtime_service import PrototypeRuntimeService
from agent.src.services.training.runtime_compatibility import (
    validate_live_agent_stored_event_runtime,
)
from shared.src.contracts.training_contracts import TrainingTaskPayload


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
RoundClientFactoryDep = Annotated[RoundClientFactory, Depends(get_round_client_factory)]
FederationRuntimeFactoryDep = Annotated[
    FederationRuntimeServiceFactory,
    Depends(get_federation_runtime_service_factory),
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
    repo: ScoredEventRepoDep,
    proto_service: ProtoServiceDep,
    round_client_factory: RoundClientFactoryDep,
    runtime_factory: FederationRuntimeFactoryDep,
) -> RunCurrentTaskResponse:
    """현재 active task를 읽어 로컬 학습을 실행하고 update를 업로드한다.

    학습 예시가 없으면 INSUFFICIENT_EXAMPLES 상태를 반환한다.
    active round 자체가 없으면 NO_ACTIVE_TASK 상태를 반환한다.
    두 경우 모두 200 OK로 응답하고 status 필드로 구분한다.
    """
    try:
        round_client = round_client_factory(request.server_base_url)
        task_payload = round_client.fetch_current_task()
        if task_payload is None:
            return RunCurrentTaskResponse(
                status="no_active_task",
                message="현재 active round 또는 open task가 없습니다.",
            )
        try:
            validate_live_agent_stored_event_runtime(task_payload)
        except ValueError as error:
            return RunCurrentTaskResponse(
                status="unsupported_runtime",
                round_id=task_payload.round_id,
                task_id=task_payload.task_id,
                message=str(error),
            )

        stored_events = repo.get_recent_stored(days=request.scored_event_days)
        try:
            active_pack = proto_service.get_active_pack()
        except FileNotFoundError:
            training_examples = ()
        else:
            scoring_service = ScoringService.from_objective_config(
                task_payload.objective_config
            )
            training_example_service = TrainingExampleService.from_objective_config(
                task_payload.objective_config
            )
            training_examples = (
                training_example_service.build_examples_from_stored_events(
                    StoredEventTrainingExampleBuildRequest(
                        stored_events=stored_events,
                        prototype_pack=active_pack,
                        scoring_service=scoring_service,
                    )
                )
            )

        service = runtime_factory(request.server_base_url)
        result: FederationRunResult = service.run_current_task(
            training_examples=training_examples,
            model_manifest=None,
            agent_id=request.agent_id,
            task_payload=task_payload,
        )
        return RunCurrentTaskResponse(
            status=str(result.status),
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
