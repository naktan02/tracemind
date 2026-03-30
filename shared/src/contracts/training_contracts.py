"""학습 task/update/feedback payload와 직렬화 유틸리티.

이 모듈은 FL orchestration 레벨에서 주고받는 envelope 계약을 정의한다.

구분:

1. `TrainingTaskPayload`
   - 서버가 agent에 내려주는 "이번 라운드에서 어떻게 학습할지" 지시문
2. `TrainingUpdateEnvelopePayload`
   - agent가 서버에 올리는 update 메타데이터 봉투
3. `DecisionFeedbackSignalPayload`
   - 로컬에서 생성된 feedback/pseudo-label/사용자 신호 단위 계약
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

TrainingConfigScalar = str | int | float | bool


class TrainingObjectiveConfigPayload(BaseModel):
    """학습 objective 관련 payload.

    - `loss`: 어떤 local training objective를 쓸지 식별자
    - `confidence_threshold`: pseudo-label 채택 최소 confidence
    - `margin_threshold`: top1-top2 차이 최소값
    - `extras`: family별 추가 하이퍼파라미터 확장 슬롯
    """

    model_config = ConfigDict(extra="forbid")

    loss: str = Field(
        default="diagonal_scale_heuristic",
        description="Local training objective 식별자.",
    )
    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pseudo-label를 채택하기 위한 최소 confidence.",
    )
    margin_threshold: float | None = Field(
        default=None,
        description="Top1과 top2 score 차이의 최소값.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="Objective family별 추가 하이퍼파라미터 확장 슬롯.",
    )


class TrainingSelectionPolicyPayload(BaseModel):
    """로컬 학습 예시 선택 정책 payload.

    - `max_examples`: 한 라운드에서 최대 몇 개 예시를 반영할지
    - `require_feedback`: 명시적 feedback가 있는 예시만 허용할지
    - `extras`: 선택 정책별 추가 규칙 확장 슬롯
    """

    model_config = ConfigDict(extra="forbid")

    max_examples: int | None = Field(
        default=None,
        ge=0,
        description="한 라운드에서 반영할 로컬 예시의 최대 개수.",
    )
    require_feedback: bool | None = Field(
        default=None,
        description="명시적 feedback가 있는 예시만 학습에 반영할지 여부.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="Selection policy별 추가 규칙 확장 슬롯.",
    )


class TrainingTaskPayload(BaseModel):
    """중앙이 로컬에 배포하는 학습 작업 payload.

    이 payload는 "어떤 모델 revision을 기준으로, 어떤 학습 규칙으로,
    언제까지 로컬 update를 만들어야 하는가"를 정의한다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(description="Payload contract 버전.")
    task_id: str = Field(description="이 학습 task의 고유 식별자.")
    round_id: str = Field(description="이 task가 속한 FL round 식별자.")
    model_id: str = Field(description="학습 대상 backbone/model 식별자.")
    model_revision: str = Field(
        description="로컬 학습이 기준으로 삼아야 할 현재 revision."
    )
    task_type: str = Field(description="학습 task 유형 식별자.")
    training_scope: str = Field(description="예: adapter_only 같은 학습 범위.")
    local_epochs: int = Field(ge=1, description="로컬 데이터 반복 횟수.")
    batch_size: int = Field(ge=1, description="로컬 학습 배치 크기.")
    learning_rate: float = Field(gt=0.0, description="로컬 optimizer 학습률.")
    max_steps: int = Field(ge=1, description="로컬에서 허용되는 최대 update step 수.")
    objective_config: TrainingObjectiveConfigPayload = Field(
        description="로컬 objective와 채택 threshold 설정."
    )
    selection_policy: TrainingSelectionPolicyPayload = Field(
        description="로컬 예시 선택 규칙."
    )
    deadline_at: datetime | None = Field(
        default=None,
        description="이 task가 유효한 마감 시각.",
    )
    gradient_clip_norm: float | None = Field(
        default=None,
        gt=0.0,
        description="로컬 update의 norm clipping 상한.",
    )
    min_required_examples: int | None = Field(
        default=None,
        ge=1,
        description="Update를 만들기 위해 필요한 최소 accepted example 수.",
    )
    secure_aggregation_required: bool = Field(
        default=False,
        description="중앙 업로드 시 secure aggregation이 필수인지 여부.",
    )
    notes: str | None = Field(
        default=None,
        description="운영 메모 또는 디버깅용 설명.",
    )


class TrainingUpdateEnvelopePayload(BaseModel):
    """로컬 agent가 중앙에 보내는 update envelope payload.

    실제 파라미터 delta는 `payload_ref`가 가리키는 파일에 있고,
    이 envelope은 round/task/revision/metrics 같은 메타데이터를 담는다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(description="Payload contract 버전.")
    update_id: str = Field(description="이 update envelope의 고유 식별자.")
    round_id: str = Field(description="이 update가 속한 FL round 식별자.")
    task_id: str = Field(description="어느 training task의 결과인지 식별자.")
    model_id: str = Field(description="학습 대상 backbone/model 식별자.")
    base_model_revision: str = Field(
        description="이 update가 계산된 기준 shared adapter revision."
    )
    training_scope: str = Field(description="예: adapter_only 같은 학습 범위.")
    payload_ref: str = Field(
        description="실제 adapter update payload 파일 경로 또는 참조값."
    )
    payload_format: str = Field(
        description="예: diagonal_scale_update 같은 update payload 포맷 식별자."
    )
    example_count: int = Field(
        ge=0,
        description="실제 update에 반영된 로컬 예시 수.",
    )
    client_metrics: dict[str, float] = Field(
        description="로컬 측 confidence, margin, loss 등의 요약 metric."
    )
    created_at: datetime | None = Field(
        default=None,
        description="Agent가 envelope을 기록한 UTC 시각.",
    )
    clipped: bool | None = Field(
        default=None,
        description="Privacy/safety guard가 clipping을 적용했는지 여부.",
    )
    dp_applied: bool | None = Field(
        default=None,
        description="Differential Privacy noise가 적용됐는지 여부.",
    )
    checksum: str | None = Field(
        default=None,
        description="Payload 무결성 검사용 체크섬.",
    )
    notes: str | None = Field(
        default=None,
        description="운영 메모 또는 디버깅용 설명.",
    )


class DecisionFeedbackSignalPayload(BaseModel):
    """로컬 학습에 사용하는 feedback signal payload.

    사용자 피드백, pseudo-label, 후속 판단 결과 같은 신호를 하나의 공통
    contract로 표현한다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(description="Payload contract 버전.")
    signal_id: str = Field(description="Feedback signal 고유 식별자.")
    signal_type: str = Field(
        description="예: pseudo_label, clinician_feedback 같은 신호 유형."
    )
    label: str = Field(description="이 signal이 가리키는 category/label.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="이 signal의 신뢰도 또는 확신도.",
    )
    occurred_at: datetime = Field(description="Signal이 발생한 UTC 시각.")
    source_event_ref: str | None = Field(
        default=None,
        description="원본 이벤트나 문장과 연결하는 참조값.",
    )
    task_context: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="Signal이 생성된 task/round 문맥 정보.",
    )
    notes: str | None = Field(
        default=None,
        description="운영 메모 또는 추가 설명.",
    )


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_training_task_payload(path: Path) -> TrainingTaskPayload:
    """JSON 파일에서 training task payload를 읽는다."""
    return TrainingTaskPayload.model_validate_json(path.read_text(encoding="utf-8"))


def dump_training_task_payload(path: Path, payload: TrainingTaskPayload) -> None:
    """training task payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_training_update_envelope_payload(
    path: Path,
) -> TrainingUpdateEnvelopePayload:
    """JSON 파일에서 update envelope payload를 읽는다."""
    return TrainingUpdateEnvelopePayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_training_update_envelope_payload(
    path: Path,
    payload: TrainingUpdateEnvelopePayload,
) -> None:
    """update envelope payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_decision_feedback_signal_payload(
    path: Path,
) -> DecisionFeedbackSignalPayload:
    """JSON 파일에서 feedback signal payload를 읽는다."""
    return DecisionFeedbackSignalPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_decision_feedback_signal_payload(
    path: Path,
    payload: DecisionFeedbackSignalPayload,
) -> None:
    """feedback signal payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)
