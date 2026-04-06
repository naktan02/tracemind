"""학습 라우터."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
    StoredScoredEvent,
)
from agent.src.services.federation.round_client import RoundClient
from agent.src.services.federation.runtime_service import (
    FederationRunResult,
    FederationRuntimeService,
)

from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.prototype.runtime_service import PrototypeRuntimeService
from agent.src.services.training.local_training_service import EmbeddedTrainingExample
from shared.src.contracts.prototype_contracts import extract_category_prototypes
from shared.src.contracts.training_contracts import TrainingTaskPayload
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.vector_adapter_state import VectorAdapterState


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


@router.post(
    "/run-current-task",
    response_model=RunCurrentTaskResponse,
    status_code=status.HTTP_200_OK,
)
def run_current_task(request: RunCurrentTaskRequest) -> RunCurrentTaskResponse:
    """현재 active task를 읽어 로컬 학습을 실행하고 update를 업로드한다.

    학습 예시가 없으면 INSUFFICIENT_EXAMPLES 상태를 반환한다.
    active round 자체가 없으면 NO_ACTIVE_TASK 상태를 반환한다.
    두 경우 모두 200 OK로 응답하고 status 필드로 구분한다.
    """
    try:
        # stored events 로딩 (base_embedding 포함)
        repo = ScoredEventRepository()
        stored_events = repo.get_recent_stored(days=request.scored_event_days)
        training_examples = _build_training_examples(stored_events)

        client = RoundClient(server_base_url=request.server_base_url)
        service = FederationRuntimeService(round_client=client)
        result: FederationRunResult = service.run_current_task(
            training_examples=training_examples,
            model_manifest=_get_placeholder_manifest(),
            agent_id=request.agent_id,
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
def get_training_status(server_base_url: str) -> TrainingStatusResponse:
    """현재 active round/task 조회."""
    try:
        client = RoundClient(server_base_url=server_base_url)
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


def _get_placeholder_manifest():
    """ModelManifest placeholder.

    TODO(Phase-3 완료): 이 함수를 제거하고 agent 로컈 manifest 저장소에서 읽는다.
    제거 조건: LocalManifestRepository.get_active() 같은 인터페이스가
              구현되고 run_current_task가 올바른 model_revision을 넘길 때.
    """
    from shared.src.domain.entities.artifacts.model_manifest import ModelManifest

    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id="placeholder",
        model_revision="rev_placeholder",
        published_at=datetime.now(tz=timezone.utc),
        artifact_kind="embedding",
        prototype_version="proto_placeholder",
        training_scope="adapter_only",
        training_enabled=True,
    )


def _build_training_examples(stored: list[StoredScoredEvent]) -> tuple[EmbeddedTrainingExample, ...]:
    """StoredScoredEvent 목록을 EmbeddedTrainingExample로 변환한다.

    프로토타입과 스코어링은 저장된 base_embedding을 기준으로 현재 active 프로토타입쿉으로 다시 실행한다.
    adapter state가 없으면 identity (scale=1.0) 를 사용한다.
    """
    # embedding이 저장된 항목만 사용
    usable = [
        s for s in stored
        if s.base_embedding is not None and len(s.base_embedding) > 0
    ]
    if not usable:
        return ()

    # 현재 active prototype 로딩
    try:
        proto_service = PrototypeRuntimeService()
        active_pack = proto_service.get_active_pack()
        prototypes = extract_category_prototypes(active_pack)
    except FileNotFoundError:
        # 로컈에 prototype이 없으면 빈 반환
        return ()

    # 임베딩 차원으로 identity adapter 생성
    embedding_dim = len(usable[0].base_embedding)  # type: ignore[arg-type]
    adapter_state = VectorAdapterState.identity(
        model_id="local",
        model_revision="local",
        training_scope="adapter_only",
        embedding_dim=embedding_dim,
        updated_at=datetime.now(tz=timezone.utc),
    )

    scoring_service = ScoringService()
    examples: list[EmbeddedTrainingExample] = []

    for stored_item in usable:
        base_emb = stored_item.base_embedding  # type: ignore[assignment]
        adapted_emb = adapter_state.apply(base_emb)
        category_scores = scoring_service.score(adapted_emb, prototypes)

        rescored_event = ScoredEvent(
            query_id=stored_item.scored_event.query_id,
            occurred_at=stored_item.scored_event.occurred_at,
            translated_text=stored_item.scored_event.translated_text,
            embedding_model_id=stored_item.scored_event.embedding_model_id,
            translation_model_id=stored_item.scored_event.translation_model_id,
            category_scores=category_scores,
        )
        examples.append(
            EmbeddedTrainingExample(
                scored_event=rescored_event,
                embedding=adapted_emb,
                base_embedding=base_emb,
            )
        )

    return tuple(examples)
