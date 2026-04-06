"""Agent FL 참여 orchestration 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from agent.src.services.federation.round_client import RoundClient
from agent.src.services.training.local_training_service import (
    EmbeddedTrainingExample,
    LocalTrainingResult,
    LocalTrainingService,
)
from shared.src.contracts.training_contracts import (
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest
from shared.src.domain.entities.training.training_task import TrainingTask
from shared.src.domain.entities.training.training_task_config import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
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
        model_manifest: ModelManifest,
    ) -> FederationRunResult:
        """현재 active task를 읽어 로컬 학습을 실행하고 결과를 업로드한다.

        학습에 필요한 EmbeddedTrainingExample은 호출자가 준비해서 넘긴다.
        서버에 active round가 없거나 task가 없으면 NO_ACTIVE_TASK를 반환한다.
        """
        task_payload = self.round_client.fetch_current_task()
        if task_payload is None:
            return FederationRunResult(
                status=FederationRunStatus.NO_ACTIVE_TASK,
                message="현재 active round 또는 open task가 없습니다.",
            )

        round_id = task_payload.round_id
        task_id = task_payload.task_id

        if task_id in self._completed_task_ids:
            return FederationRunResult(
                status=FederationRunStatus.ALREADY_COMPLETED,
                round_id=round_id,
                task_id=task_id,
                message=f"이미 완료한 task입니다: {task_id}",
            )

        training_task = _task_from_payload(task_payload)
        local_result: LocalTrainingResult = self.local_training_service.run_task(
            training_examples=training_examples,
            training_task=training_task,
            model_manifest=model_manifest,
            created_at=datetime.now(tz=timezone.utc),
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
        raise ValueError("update_envelope이 없는 result를 payload로 변환할 수 없습니다.")
    return TrainingUpdateEnvelopePayload(
        schema_version=envelope.schema_version,
        update_id=envelope.update_id,
        round_id=envelope.round_id,
        task_id=envelope.task_id,
        model_id=envelope.model_id,
        base_model_revision=envelope.base_model_revision,
        training_scope=envelope.training_scope,
        payload_ref=envelope.payload_ref,
        payload_format=envelope.payload_format,
        example_count=envelope.example_count,
        client_metrics=envelope.client_metrics,
        created_at=envelope.created_at,
        clipped=envelope.clipped,
        dp_applied=envelope.dp_applied,
        agent_id=envelope.agent_id,
    )


def _task_from_payload(payload: TrainingTaskPayload) -> TrainingTask:
    """TrainingTaskPayload를 domain TrainingTask로 변환한다."""
    return TrainingTask(
        schema_version=payload.schema_version,
        task_id=payload.task_id,
        round_id=payload.round_id,
        model_id=payload.model_id,
        model_revision=payload.model_revision,
        task_type=payload.task_type,
        training_scope=payload.training_scope,
        local_epochs=payload.local_epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_steps=payload.max_steps,
        objective_config=TrainingObjectiveConfig(
            loss=payload.objective_config.loss,
            confidence_threshold=payload.objective_config.confidence_threshold,
            margin_threshold=payload.objective_config.margin_threshold,
            score_policy_name=payload.objective_config.score_policy_name,
            score_top_k=payload.objective_config.score_top_k,
            acceptance_policy_name=payload.objective_config.acceptance_policy_name,
            privacy_guard_name=payload.objective_config.privacy_guard_name,
            extras=dict(payload.objective_config.extras),
        ),
        selection_policy=TrainingSelectionPolicy(
            max_examples=payload.selection_policy.max_examples,
            require_feedback=payload.selection_policy.require_feedback,
            extras=dict(payload.selection_policy.extras),
        ),
        deadline_at=payload.deadline_at,
        gradient_clip_norm=payload.gradient_clip_norm,
        min_required_examples=payload.min_required_examples,
        secure_aggregation_required=payload.secure_aggregation_required,
        notes=payload.notes,
    )
