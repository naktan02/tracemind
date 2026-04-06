"""학습 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict

from agent.src.services.federation.runtime_service import (
    FederationRunResult,
    FederationRunStatus,
    FederationRuntimeService,
)
from agent.src.services.federation.round_client import RoundClient
from shared.src.contracts.training_contracts import TrainingTaskPayload


class RunCurrentTaskRequest(BaseModel):
    """run-current-task API 요청."""

    model_config = ConfigDict(extra="forbid")

    server_base_url: str


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

    NOTE: 현재 구현은 training_examples를 외부에서 주입받지 않아
    INSUFFICIENT_EXAMPLES가 항상 반환된다. 실제 로컬 데이터 파이프라인
    연결은 Phase 2 이후 단계에서 추가된다.
    """
    try:
        client = RoundClient(server_base_url=request.server_base_url)
        service = FederationRuntimeService(round_client=client)
        # TODO(Phase-3): inference pipeline 연결 후 아래 두 줄을 교체한다.
        #   training_examples ← InferencePipelineService에서 scored events를 가져옴
        #   model_manifest    ← agent 로컬 manifest 저장소에서 active manifest를 읽음
        #   제거 조건: FederationRuntimeService.run_current_task()가 실제 예시를 받아
        #             UPLOADED 상태를 반환할 수 있을 때.
        result: FederationRunResult = service.run_current_task(
            training_examples=(),
            model_manifest=_get_placeholder_manifest(),
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

    TODO(Phase-3): 이 함수를 제거하고 agent 로컬 manifest 저장소
    (예: PrototypePackRepository 또는 별도 ManifestRepository)에서
    현재 active manifest를 읽는 것으로 교체한다.
    제거 조건: LocalManifestRepository.get_active() 같은 인터페이스가
              구현되고 run_current_task가 올바른 model_revision을 넘길 때.
    """
    from datetime import datetime, timezone

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
