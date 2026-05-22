"""학습 task/update/feedback payload와 직렬화 유틸리티.

이 모듈은 FL orchestration 레벨에서 주고받는 envelope 계약을 정의한다.

구분:

1. `TrainingTaskPayload`
   - 서버가 agent에 내려주는 "이번 라운드에서 어떻게 학습할지" 지시문
2. `TrainingUpdateEnvelopePayload`
   - 서버가 수락/저장한 update 메타데이터 봉투
3. `TrainingUpdateSubmissionPayload`
   - agent가 서버에 제출하는 envelope + inline update payload
4. `DecisionFeedbackSignalPayload`
   - 로컬에서 생성된 feedback/pseudo-label/사용자 신호 단위 계약
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.registry import (
    parse_shared_adapter_update_payload,
)
from shared.src.contracts.secure_aggregation_contracts import (
    SecureAggregationConfigPayload,
    SecureAggregationSubmissionPayload,
)
from shared.src.contracts.training_objective_contracts import (
    TrainingConfigScalar,
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
)

from .common_types import TrainingScope, TrainingTaskType, normalize_training_task_type

TRAINING_TASK_V1 = "training_task.v1"
TRAINING_UPDATE_ENVELOPE_V1 = "training_update_envelope.v1"
TRAINING_UPDATE_SUBMISSION_V1 = "training_update_submission.v1"
DECISION_FEEDBACK_SIGNAL_V1 = "decision_feedback_signal.v1"
TrainingTaskSchemaVersion: TypeAlias = Literal["training_task.v1"]
TrainingUpdateEnvelopeSchemaVersion: TypeAlias = Literal["training_update_envelope.v1"]
TrainingUpdateSubmissionSchemaVersion: TypeAlias = Literal[
    "training_update_submission.v1"
]
DecisionFeedbackSignalSchemaVersion: TypeAlias = Literal["decision_feedback_signal.v1"]


class FeedbackSignalType(StrEnum):
    """로컬 학습 signal taxonomy."""

    PSEUDO_LABEL = "pseudo_label"
    SELF_REPORT = "self_report"
    SUPPORT_ACTION = "support_action"
    DELAYED_OUTCOME = "delayed_outcome"


class ClientMetricKeys:
    """TrainingUpdateEnvelopePayload.client_metrics의 표준 키 상수.

    client_metrics는 dict[str, float]로 열려 있지만,
    서버 aggregation과 q-합의 알고리즘이 사용하는 표준 키를 여기 선언한다.
    사실상 타입 제약이므로, 동작을 추가하지 않고 key 저장소로만 사용한다.
    """

    # pseudo-label selection 스테이지에서 종합 컨피던스 평균
    MEAN_CONFIDENCE = "mean_confidence"
    # top1 접수와 top2 접수 차이의 평균
    MEAN_MARGIN = "mean_margin"
    # 주어진 scored events 중 선택된 비율
    ACCEPTED_RATIO = "accepted_ratio"
    # privacy guard 적용 후 delta 벡터의 L2 norm
    DELTA_L2_NORM = "delta_l2_norm"
    # update에 반영된 실제 예시 수 (float로 저장)
    SELECTED_EXAMPLES = "selected_examples"


class TrainingTaskPayload(BaseModel):
    """중앙이 로컬에 배포하는 학습 작업 payload.

    이 payload는 "어떤 모델 revision을 기준으로, 어떤 학습 규칙으로,
    언제까지 로컬 update를 만들어야 하는가"를 정의한다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: TrainingTaskSchemaVersion = Field(
        default=TRAINING_TASK_V1,
        description="Payload contract 버전.",
    )
    task_id: str = Field(description="이 학습 task의 고유 식별자.")
    round_id: str = Field(description="이 task가 속한 FL round 식별자.")
    model_id: str = Field(description="학습 대상 backbone/model 식별자.")
    model_revision: str = Field(
        description="로컬 학습이 기준으로 삼아야 할 현재 revision."
    )
    task_type: TrainingTaskType = Field(description="학습 task 유형 식별자.")
    training_scope: TrainingScope = Field(
        description="예: adapter_only 같은 학습 범위."
    )
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
    secure_aggregation: SecureAggregationConfigPayload = Field(
        default_factory=SecureAggregationConfigPayload,
        description="중앙 업로드 시 요구되는 secure aggregation/encryption 설정.",
    )
    notes: str | None = Field(
        default=None,
        description="운영 메모 또는 디버깅용 설명.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_secure_aggregation_flag(
        cls,
        source: object,
    ) -> object:
        if not isinstance(source, Mapping):
            return source
        data = dict(source)
        if "task_type" in data:
            data["task_type"] = normalize_training_task_type(data["task_type"])
        legacy_required = data.pop("secure_aggregation_required", None)
        secure_aggregation = data.get("secure_aggregation")
        if secure_aggregation is None:
            if legacy_required is not None:
                data["secure_aggregation"] = {"required": bool(legacy_required)}
            return data
        if legacy_required is None or not isinstance(secure_aggregation, Mapping):
            return data
        normalized = dict(secure_aggregation)
        normalized.setdefault("required", bool(legacy_required))
        data["secure_aggregation"] = normalized
        return data

    @property
    def secure_aggregation_required(self) -> bool:
        """구버전 bool 플래그와의 호환을 위한 derived alias."""

        return self.secure_aggregation.required


class TrainingUpdateEnvelopePayload(BaseModel):
    """서버가 수락/저장한 update envelope payload.

    이 envelope은 round/task/revision/metrics 같은 메타데이터와 서버가
    저장한 update payload 참조를 담는다. Agent가 서버에 제출할 때는
    `TrainingUpdateSubmissionPayload`로 실제 update payload를 함께 보낸다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: TrainingUpdateEnvelopeSchemaVersion = Field(
        default=TRAINING_UPDATE_ENVELOPE_V1,
        description="Payload contract 버전.",
    )
    update_id: str = Field(description="이 update envelope의 고유 식별자.")
    round_id: str = Field(description="이 update가 속한 FL round 식별자.")
    task_id: str = Field(description="어느 training task의 결과인지 식별자.")
    model_id: str = Field(description="학습 대상 backbone/model 식별자.")
    base_model_revision: str = Field(
        description="이 update가 계산된 기준 shared adapter revision."
    )
    training_scope: TrainingScope = Field(
        description="예: adapter_only 같은 학습 범위."
    )
    payload_ref: str = Field(
        description=(
            "서버가 저장한 adapter update payload 참조. "
            "서버 밖에서는 파일 경로로 해석하지 않는 opaque ref다. "
            "pre-submission 단계의 agent-local ref는 서버가 신뢰하지 않는다."
        )
    )
    payload_format: str = Field(
        description="예: diagonal_scale_update 같은 update payload 포맷 식별자."
    )
    example_count: int = Field(
        ge=0,
        description="실제 update에 반영된 로컬 예시 수.",
    )
    client_metrics: dict[str, float] = Field(
        description=(
            "로컬 측 학습 품질 요약 metric. "
            "표준 키는 이 모듈의 ClientMetricKeys 참고. "
            "기본 키: mean_confidence, mean_margin, accepted_ratio, "
            "delta_l2_norm, selected_examples. "
            "추가 키는 하위 호환성을 유지하며 자유롭게 확장 가능하다."
        )
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
    agent_id: str | None = Field(
        default=None,
        description=(
            "Agent의 pseudonymous 식별자. "
            "실제 사용자 신원이 아닌 기기가 직접 생성한 UUID. "
            "q-합의 알고리즘에서 per-agent 신뢰도 추적과 "
            "한 round 내 중복 제출 차단에 사용한다. "
            "None이면 완전 익명 모드."
        ),
    )
    secure_aggregation: SecureAggregationSubmissionPayload | None = Field(
        default=None,
        description="secure aggregation/encryption 제출 메타데이터.",
    )
    notes: str | None = Field(
        default=None,
        description="운영 메모 또는 디버깅용 설명.",
    )


class TrainingUpdateSubmissionPayload(BaseModel):
    """Agent가 서버에 제출하는 update 요청 payload.

    `envelope`은 metadata를 담고, `update_payload`는 실제 adapter update를
    inline으로 담는다. 서버는 수신한 update payload를 server-owned storage에
    저장한 뒤 envelope의 `payload_ref`를 서버 참조로 덮어쓴다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: TrainingUpdateSubmissionSchemaVersion = Field(
        default=TRAINING_UPDATE_SUBMISSION_V1,
        description="Update submission contract 버전.",
    )
    envelope: TrainingUpdateEnvelopePayload = Field(
        description="Round/task/update metadata envelope."
    )
    update_payload: SerializeAsAny[SharedAdapterUpdatePayload] = Field(
        description="실제 shared adapter update payload."
    )

    @field_validator("update_payload", mode="before")
    @classmethod
    def _parse_update_payload(cls, value: object) -> SharedAdapterUpdatePayload:
        if isinstance(value, SharedAdapterUpdatePayload):
            return value
        if isinstance(value, Mapping):
            return parse_shared_adapter_update_payload(value)
        raise ValueError("update_payload must be a shared adapter update payload.")

    @model_validator(mode="after")
    def _validate_envelope_alignment(self) -> "TrainingUpdateSubmissionPayload":
        update_payload = self.update_payload
        envelope = self.envelope
        if update_payload.model_id != envelope.model_id:
            raise ValueError("Submission update_payload.model_id must match envelope.")
        if update_payload.base_model_revision != envelope.base_model_revision:
            raise ValueError(
                "Submission update_payload.base_model_revision must match envelope."
            )
        if update_payload.training_scope != envelope.training_scope:
            raise ValueError(
                "Submission update_payload.training_scope must match envelope."
            )
        if update_payload.example_count != envelope.example_count:
            raise ValueError(
                "Submission update_payload.example_count must match envelope."
            )
        accepted_formats = _accepted_update_payload_formats(update_payload)
        if accepted_formats and str(envelope.payload_format) not in accepted_formats:
            raise ValueError(
                "Submission envelope.payload_format must match update_payload "
                f"accepted format(s): {accepted_formats}."
            )
        return self


def _accepted_update_payload_formats(
    update_payload: SharedAdapterUpdatePayload,
) -> tuple[str, ...]:
    accepted_formats = tuple(
        str(payload_format)
        for payload_format in getattr(
            update_payload,
            "accepted_update_payload_formats",
            (),
        )
        if str(payload_format).strip()
    )
    canonical_format = getattr(
        update_payload,
        "canonical_update_payload_format",
        None,
    )
    if canonical_format is None or str(canonical_format) in accepted_formats:
        return accepted_formats
    return (*accepted_formats, str(canonical_format))


class DecisionFeedbackSignalPayload(BaseModel):
    """로컬 학습에 사용하는 feedback signal payload.

    사용자 피드백, pseudo-label, 후속 판단 결과 같은 신호를 하나의 공통
    contract로 표현한다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: DecisionFeedbackSignalSchemaVersion = Field(
        default=DECISION_FEEDBACK_SIGNAL_V1,
        description="Payload contract 버전.",
    )
    signal_id: str = Field(description="Feedback signal 고유 식별자.")
    signal_type: FeedbackSignalType = Field(
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


def load_training_update_submission_payload(
    path: Path,
) -> TrainingUpdateSubmissionPayload:
    """JSON 파일에서 update submission payload를 읽는다."""
    return TrainingUpdateSubmissionPayload.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dump_training_update_submission_payload(
    path: Path,
    payload: TrainingUpdateSubmissionPayload,
) -> None:
    """update submission payload를 JSON 파일로 기록한다."""
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


# ------------------------------------------------------------------ #
# Factory 함수                                                         #
# ------------------------------------------------------------------ #


def make_training_update_envelope(
    *,
    round_id: str,
    task_id: str,
    model_id: str,
    base_model_revision: str,
    payload_ref: str,
    payload_format: str,
    example_count: int,
    client_metrics: dict[str, float],
    update_id: str | None = None,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    agent_id: str | None = None,
    secure_aggregation: SecureAggregationSubmissionPayload | None = None,
    clipped: bool = False,
    dp_applied: bool = False,
    notes: str | None = None,
    created_at: datetime | None = None,
) -> TrainingUpdateEnvelopePayload:
    """TrainingUpdateEnvelopePayload를 만드는 표준 factory.

    round_id, task_id, model 식별자, payload 참조, 예시 수, 메트릭만
    지정하면 나머지는 기본값으로 채워진다. 서버에 제출할 때는 이 envelope을
    `TrainingUpdateSubmissionPayload`의 `envelope` 필드로 감싼다.

    >>> import uuid
    >>> env = make_training_update_envelope(
    ...     round_id="round_abc",
    ...     task_id="task_abc",
    ...     model_id="bg-m3",
    ...     base_model_revision="rev_001",
    ...     payload_ref="client-submission::delta_001",
    ...     payload_format="diagonal_scale_update",
    ...     example_count=10,
    ...     client_metrics={"mean_confidence": 0.85},
    ... )
    """
    import uuid as _uuid

    return TrainingUpdateEnvelopePayload(
        schema_version=TRAINING_UPDATE_ENVELOPE_V1,
        update_id=update_id or str(_uuid.uuid4()),
        round_id=round_id,
        task_id=task_id,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        payload_ref=payload_ref,
        payload_format=payload_format,
        example_count=example_count,
        client_metrics=client_metrics,
        created_at=created_at or datetime.now(tz=timezone.utc),
        clipped=clipped,
        dp_applied=dp_applied,
        agent_id=agent_id,
        secure_aggregation=secure_aggregation,
        notes=notes,
    )


def make_training_update_submission(
    *,
    envelope: TrainingUpdateEnvelopePayload,
    update_payload: SharedAdapterUpdatePayload,
) -> TrainingUpdateSubmissionPayload:
    """inline update payload를 담은 server submission payload를 만든다."""

    return TrainingUpdateSubmissionPayload(
        schema_version=TRAINING_UPDATE_SUBMISSION_V1,
        envelope=envelope,
        update_payload=update_payload,
    )


TrainingObjectiveConfig = TrainingObjectiveConfigPayload
SecureAggregationConfig = SecureAggregationConfigPayload
SecureAggregationSubmission = SecureAggregationSubmissionPayload
TrainingSelectionPolicy = TrainingSelectionPolicyPayload
TrainingTask = TrainingTaskPayload
TrainingUpdateEnvelope = TrainingUpdateEnvelopePayload
TrainingUpdateSubmission = TrainingUpdateSubmissionPayload
DecisionFeedbackSignal = DecisionFeedbackSignalPayload
