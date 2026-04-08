"""Agent FL 참여 orchestration 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from agent.src.services.federation.round_client import RoundClient
from agent.src.services.training.local_training_service import (
    LocalTrainingResult,
    LocalTrainingService,
)
from agent.src.services.training.training_example_models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.model_contracts import ModelManifest, make_embedding_manifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)


class FederationRunStatus(StrEnum):
    """run-current-task 실행 결과 상태."""

    # active round/task가 없어서 실행을 건너뜀
    NO_ACTIVE_TASK = "no_active_task"
    # 채택된 예시가 부족해 update를 만들지 않음
    INSUFFICIENT_EXAMPLES = "insufficient_examples"
    # update 생성 후 서버 업로드 성공
    UPLOADED = "uploaded"
    # 이미 처리한 task라 중복 실행을 건너뜀
    ALREADY_COMPLETED = "already_completed"


@dataclass(slots=True)
class FederationRunResult:
    """run-current-task 한 번의 결과 요약."""

    status: FederationRunStatus
    round_id: str | None = None
    task_id: str | None = None
    update_id: str | None = None
    example_count: int = 0
    accepted_count: int = 0
    message: str = ""


@dataclass(slots=True)
class FederationRuntimeService:
    """server round를 읽고 local training을 실행한 뒤 update를 업로드한다.

    이 서비스의 책임은 orchestration뿐이다.
    - local training 로직: LocalTrainingService
    - server 통신: RoundClient
    학습 예시는 외부에서 주입받는다. (training_examples 파라미터)
    """

    round_client: RoundClient
    local_training_service: LocalTrainingService = field(
        default_factory=LocalTrainingService
    )
    # 이미 완료한 task_id를 기억해 중복 실행을 막는다
    _completed_task_ids: set[str] = field(default_factory=set)

    def run_current_task(
        self,
        *,
        training_examples: tuple[EmbeddedTrainingExample, ...]
        | list[EmbeddedTrainingExample],
        model_manifest: ModelManifest | None = None,
        agent_id: str | None = None,
        task_payload: TrainingTaskPayload | None = None,
    ) -> FederationRunResult:
        """현재 active task를 읽어 로컬 학습을 실행하고 결과를 업로드한다.

        학습에 필요한 EmbeddedTrainingExample은 호출자가 준비해서 넘긴다.
        서버에 active round가 없거나 task가 없으면 NO_ACTIVE_TASK를 반환한다.
        """
        effective_task_payload = task_payload or self.round_client.fetch_current_task()
        if effective_task_payload is None:
            return FederationRunResult(
                status=FederationRunStatus.NO_ACTIVE_TASK,
                message="현재 active round 또는 open task가 없습니다.",
            )

        round_id = effective_task_payload.round_id
        task_id = effective_task_payload.task_id

        if task_id in self._completed_task_ids:
            return FederationRunResult(
                status=FederationRunStatus.ALREADY_COMPLETED,
                round_id=round_id,
                task_id=task_id,
                message=f"이미 완료한 task입니다: {task_id}",
            )

        training_task = _task_from_payload(effective_task_payload)
        effective_manifest = model_manifest or _fallback_manifest_from_task(
            training_task
        )
        local_result: LocalTrainingResult = self.local_training_service.run_task(
            training_examples=training_examples,
            training_task=training_task,
            model_manifest=effective_manifest,
            created_at=datetime.now(tz=timezone.utc),
            agent_id=agent_id,
        )

        selection = local_result.selection_result
        if local_result.update_envelope is None:
            return FederationRunResult(
                status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
                round_id=round_id,
                task_id=task_id,
                example_count=selection.total_count,
                accepted_count=selection.accepted_count,
                message=(
                    f"채택된 예시 부족: "
                    f"{selection.accepted_count}/{selection.total_count}"
                ),
            )

        envelope_payload = _envelope_to_payload(local_result)
        self.round_client.upload_update(round_id, envelope_payload)
        self._completed_task_ids.add(task_id)

        return FederationRunResult(
            status=FederationRunStatus.UPLOADED,
            round_id=round_id,
            task_id=task_id,
            update_id=local_result.update_envelope.update_id,
            example_count=selection.total_count,
            accepted_count=selection.accepted_count,
            message="update 업로드 완료.",
        )

    def clear_completed(self) -> None:
        """완료 task 기록을 초기화한다. 새 round 시작 시 호출하면 된다."""
        self._completed_task_ids.clear()


def _envelope_to_payload(result: LocalTrainingResult) -> TrainingUpdateEnvelopePayload:
    """LocalTrainingResult에서 서버 업로드용 payload를 만든다."""
    envelope = result.update_envelope
    if envelope is None:
        raise ValueError(
            "update_envelope이 없는 result를 payload로 변환할 수 없습니다."
        )
    return envelope


def _task_from_payload(payload: TrainingTaskPayload) -> TrainingTask:
    """TrainingTaskPayload를 domain TrainingTask로 변환한다."""
    return payload


def _fallback_manifest_from_task(task: TrainingTask) -> ModelManifest:
    """task payload만 있을 때 local training이 쓸 최소 manifest를 만든다."""
    return make_embedding_manifest(
        model_id=task.model_id,
        model_revision=task.model_revision,
        prototype_version=f"task::{task.round_id}",
        artifact_ref=f"training_task::{task.task_id}",
        published_at=datetime.now(tz=timezone.utc),
        training_scope=task.training_scope,
        compatible_task_types=(task.task_type,),
    )
