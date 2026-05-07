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
from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from shared.src.contracts.adapter_contracts import (
    SharedAdapterUpdatePayload,
    parse_shared_adapter_update_payload,
)

from .common_types import TrainingScope, TrainingTaskType

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


class UpdatePayloadFormat(StrEnum):
    """Training update payload 포맷의 알려진 상수 모음."""

    DIAGONAL_SCALE_UPDATE = "diagonal_scale_update"
    CLASSIFIER_HEAD_UPDATE = "classifier_head_update"
    LORA_CLASSIFIER_UPDATE = "lora_classifier_update"
    LEGACY_VECTOR_ADAPTER_DELTA = "vector_adapter_delta"


class FeedbackSignalType(StrEnum):
    """로컬 학습 signal taxonomy."""

    PSEUDO_LABEL = "pseudo_label"
    SELF_REPORT = "self_report"
    SUPPORT_ACTION = "support_action"
    DELAYED_OUTCOME = "delayed_outcome"


TrainingConfigScalar = str | int | float | bool
DEFAULT_TRAINING_BACKEND_NAME = "diagonal_scale_heuristic"


class ClientMetricKeys:
    """TrainingUpdateEnvelopePayload.client_metrics의 표준 키 상수.

    client_metrics는 dict[str, float]로 열려 있지만,
    서버 aggregation과 q-합의 알고리즘이 사용하는 표준 키를 여기 선언한다.
    사실상 타입 제약이므로, 이 클래스를 갑율역할 필요 없다. key 저장소로만 사용할 것
    """

    # --- 하학 질 요약 ---
    # pseudo-label selection 스테이지에서 종합 컨피던스 평균
    MEAN_CONFIDENCE = "mean_confidence"
    # top1 접수와 top2 접수 차이의 평균
    MEAN_MARGIN = "mean_margin"
    # 주어진 scored events 중 선탉된 비율
    ACCEPTED_RATIO = "accepted_ratio"
    # privacy guard 적용 후 delta 벡터의 L2 norm
    DELTA_L2_NORM = "delta_l2_norm"
    # update에 반영된 실제 예시 수 (float로 저장)
    SELECTED_EXAMPLES = "selected_examples"


class TrainingObjectiveConfigPayload(BaseModel):
    """학습 objective 관련 payload.

    - `training_backend_name`: 어떤 local update backend를 쓸지 식별자
    - `algorithm_profile_name`: 논문/알고리즘 단위 조합 preset 식별자
    - `loss_name`: 학습 objective의 loss 함수 식별자
    - `confidence_threshold`: pseudo-label 채택 최소 confidence
    - `margin_threshold`: top1-top2 차이 최소값
    - `example_generation_backend_name`: 학습 예시 재구성 backend 식별자
    - `evidence_backend_name`: pseudo-label evidence 정규화 backend 식별자
    - `scorer_backend_name`: category score 계산 backend 식별자
    - `score_policy_name`: 다중 prototype score 집계 정책 식별자
    - `score_top_k`: top-k 계열 score 정책이 사용할 k 값
    - `pseudo_label_algorithm_name`: pseudo-label 후보를 평가하는
      selection 알고리즘 식별자
    - `acceptance_policy_name`: runtime compatibility가 검증할 acceptance 정책 식별자
    - `privacy_guard_name`: 로컬 update 보호 계층 식별자
    - `extras`: family별 추가 하이퍼파라미터 확장 슬롯
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    training_backend_name: str = Field(
        default=DEFAULT_TRAINING_BACKEND_NAME,
        validation_alias=AliasChoices("training_backend_name", "loss"),
        serialization_alias="training_backend_name",
        description="로컬 update backend 식별자.",
    )
    algorithm_profile_name: str | None = Field(
        default=None,
        description="논문/알고리즘 단위 objective 조합 preset 식별자.",
    )
    loss_name: str | None = Field(
        default=None,
        description=(
            "학습 objective의 loss 함수 식별자. backend 선택과는 독립적인 의미 축이다."
        ),
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
    example_generation_backend_name: str | None = Field(
        default=None,
        description="학습 예시 재구성 backend 식별자.",
    )
    evidence_backend_name: str | None = Field(
        default=None,
        description="Pseudo-label evidence 정규화 backend 식별자.",
    )
    scorer_backend_name: str | None = Field(
        default=None,
        description="카테고리 score 계산 backend 식별자.",
    )
    score_policy_name: str | None = Field(
        default=None,
        description="카테고리 내 다중 prototype score 집계 정책 식별자.",
    )
    score_top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k score 집계 정책이 사용할 k 값.",
    )
    pseudo_label_algorithm_name: str | None = Field(
        default=None,
        description="Pseudo-label 후보를 평가하는 selection 알고리즘 식별자.",
    )
    acceptance_policy_name: str | None = Field(
        default=None,
        description="Pseudo-label acceptance 정책 식별자.",
    )
    privacy_guard_name: str | None = Field(
        default=None,
        description="로컬 update 보호 계층 식별자.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="Objective family별 추가 하이퍼파라미터 확장 슬롯.",
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar] | None,
    ) -> "TrainingObjectiveConfigPayload":
        """Mapping 입력을 canonical objective config로 정규화한다."""
        if source is None:
            return cls()
        source = dict(source)
        backend_name = source.get(
            "training_backend_name",
            source.get("loss", DEFAULT_TRAINING_BACKEND_NAME),
        )
        pseudo_label_algorithm_name = _optional_str(
            source.get("pseudo_label_algorithm_name")
        )
        if pseudo_label_algorithm_name is None:
            # compatibility:
            # 과거 objective mapping은 acceptance 정책 이름을 selection 알고리즘
            # 식별자로도 재사용했다. canonical contract는 분리하되,
            # mapping 정규화 경로에서만 얇게 이어받는다.
            pseudo_label_algorithm_name = _optional_str(
                source.get("acceptance_policy_name")
            )
        return cls(
            training_backend_name=str(backend_name),
            algorithm_profile_name=_optional_str(source.get("algorithm_profile_name")),
            loss_name=_optional_str(source.get("loss_name")),
            confidence_threshold=_optional_float(source.get("confidence_threshold")),
            margin_threshold=_optional_float(source.get("margin_threshold")),
            example_generation_backend_name=_optional_str(
                source.get("example_generation_backend_name")
            ),
            evidence_backend_name=_optional_str(source.get("evidence_backend_name")),
            scorer_backend_name=_optional_str(source.get("scorer_backend_name")),
            score_policy_name=_optional_str(source.get("score_policy_name")),
            score_top_k=_optional_positive_int(source.get("score_top_k")),
            pseudo_label_algorithm_name=pseudo_label_algorithm_name,
            acceptance_policy_name=_optional_str(source.get("acceptance_policy_name")),
            privacy_guard_name=_optional_str(source.get("privacy_guard_name")),
            extras={
                key: value
                for key, value in source.items()
                if key
                not in {
                    "training_backend_name",
                    "algorithm_profile_name",
                    "loss",
                    "loss_name",
                    "confidence_threshold",
                    "margin_threshold",
                    "example_generation_backend_name",
                    "evidence_backend_name",
                    "scorer_backend_name",
                    "score_policy_name",
                    "score_top_k",
                    "pseudo_label_algorithm_name",
                    "acceptance_policy_name",
                    "privacy_guard_name",
                }
            },
        )

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        """canonical objective config를 저장/전송용 flat mapping으로 변환한다."""
        result: dict[str, TrainingConfigScalar] = {
            "training_backend_name": self.training_backend_name
        }
        if self.algorithm_profile_name is not None:
            result["algorithm_profile_name"] = self.algorithm_profile_name
        if self.loss_name is not None:
            result["loss_name"] = self.loss_name
        if self.confidence_threshold is not None:
            result["confidence_threshold"] = self.confidence_threshold
        if self.margin_threshold is not None:
            result["margin_threshold"] = self.margin_threshold
        if self.example_generation_backend_name is not None:
            result["example_generation_backend_name"] = (
                self.example_generation_backend_name
            )
        if self.evidence_backend_name is not None:
            result["evidence_backend_name"] = self.evidence_backend_name
        if self.scorer_backend_name is not None:
            result["scorer_backend_name"] = self.scorer_backend_name
        if self.score_policy_name is not None:
            result["score_policy_name"] = self.score_policy_name
        if self.score_top_k is not None:
            result["score_top_k"] = self.score_top_k
        if self.pseudo_label_algorithm_name is not None:
            result["pseudo_label_algorithm_name"] = self.pseudo_label_algorithm_name
        if self.acceptance_policy_name is not None:
            result["acceptance_policy_name"] = self.acceptance_policy_name
        if self.privacy_guard_name is not None:
            result["privacy_guard_name"] = self.privacy_guard_name
        result.update(self.extras)
        return result

    def get_component_extras(
        self,
        component_scope: str,
        *,
        legacy_keys: Collection[str] = (),
    ) -> dict[str, TrainingConfigScalar]:
        """컴포넌트 scope별 extra 파라미터를 추출한다."""

        normalized_scope = component_scope.strip()
        if not normalized_scope:
            raise ValueError("component_scope must not be empty.")
        prefix = f"{normalized_scope}."
        scoped = {
            key[len(prefix) :]: value
            for key, value in self.extras.items()
            if key.startswith(prefix)
        }
        if scoped:
            return scoped
        return {key: value for key, value in self.extras.items() if key in legacy_keys}

    @property
    def loss(self) -> str:
        """구버전 config key와의 호환을 위한 deprecated alias."""
        return self.training_backend_name


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

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar] | None,
    ) -> "TrainingSelectionPolicyPayload":
        """Mapping 입력을 canonical selection policy로 정규화한다."""
        if source is None:
            return cls()
        return cls(
            max_examples=_optional_int(source.get("max_examples")),
            require_feedback=_optional_bool(source.get("require_feedback")),
            extras={
                key: value
                for key, value in source.items()
                if key not in {"max_examples", "require_feedback"}
            },
        )

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        """canonical selection policy를 저장/전송용 flat mapping으로 변환한다."""
        result: dict[str, TrainingConfigScalar] = {}
        if self.max_examples is not None:
            result["max_examples"] = self.max_examples
        if self.require_feedback is not None:
            result["require_feedback"] = self.require_feedback
        result.update(self.extras)
        return result


class SecureAggregationConfigPayload(BaseModel):
    """학습 task가 요구하는 secure aggregation/encryption 설정."""

    model_config = ConfigDict(extra="forbid")

    required: bool = Field(
        default=False,
        description="이번 task에서 secure aggregation이 필수인지 여부.",
    )
    aggregation_backend_name: str | None = Field(
        default=None,
        description="secure aggregation backend 식별자.",
    )
    encryption_scheme_name: str | None = Field(
        default=None,
        description="예: ckks, paillier 같은 encryption scheme 식별자.",
    )
    key_ref: str | None = Field(
        default=None,
        description="암호화 컨텍스트/공개키 material 참조값.",
    )
    ciphertext_format: str | None = Field(
        default=None,
        description="예: ckks_vector_v1 같은 ciphertext 직렬화 포맷 식별자.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="secure aggregation backend별 추가 파라미터 확장 슬롯.",
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar] | None,
    ) -> "SecureAggregationConfigPayload":
        if source is None:
            return cls()
        return cls(
            required=_optional_bool(source.get("required")) or False,
            aggregation_backend_name=_optional_str(
                source.get("aggregation_backend_name")
            ),
            encryption_scheme_name=_optional_str(source.get("encryption_scheme_name")),
            key_ref=_optional_str(source.get("key_ref")),
            ciphertext_format=_optional_str(source.get("ciphertext_format")),
            extras={
                key: value
                for key, value in source.items()
                if key
                not in {
                    "required",
                    "aggregation_backend_name",
                    "encryption_scheme_name",
                    "key_ref",
                    "ciphertext_format",
                }
            },
        )

    @model_validator(mode="after")
    def _normalize_required(self) -> "SecureAggregationConfigPayload":
        if self.required:
            return self
        if (
            any(
                value is not None
                for value in (
                    self.aggregation_backend_name,
                    self.encryption_scheme_name,
                    self.key_ref,
                    self.ciphertext_format,
                )
            )
            or self.extras
        ):
            self.required = True
        return self

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        result: dict[str, TrainingConfigScalar] = {"required": self.required}
        if self.aggregation_backend_name is not None:
            result["aggregation_backend_name"] = self.aggregation_backend_name
        if self.encryption_scheme_name is not None:
            result["encryption_scheme_name"] = self.encryption_scheme_name
        if self.key_ref is not None:
            result["key_ref"] = self.key_ref
        if self.ciphertext_format is not None:
            result["ciphertext_format"] = self.ciphertext_format
        result.update(self.extras)
        return result


class SecureAggregationSubmissionPayload(BaseModel):
    """업로드된 update가 어떤 secure aggregation/encryption metadata를 따르는지."""

    model_config = ConfigDict(extra="forbid")

    aggregation_backend_name: str = Field(
        description="제출된 암호문/secure aggregation backend 식별자."
    )
    encryption_scheme_name: str | None = Field(
        default=None,
        description="예: ckks, paillier 같은 encryption scheme 식별자.",
    )
    key_ref: str | None = Field(
        default=None,
        description="사용한 키 material 또는 encryption context 참조값.",
    )
    ciphertext_format: str | None = Field(
        default=None,
        description="업로드 payload의 ciphertext 직렬화 포맷 식별자.",
    )
    extras: dict[str, TrainingConfigScalar] = Field(
        default_factory=dict,
        description="secure submission별 추가 메타데이터 확장 슬롯.",
    )


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
        return self


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


def _optional_float(value: TrainingConfigScalar | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected float-like config value, got bool.")
    return float(value)


def _optional_int(value: TrainingConfigScalar | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected int-like config value, got bool.")
    return int(value)


def _optional_bool(value: TrainingConfigScalar | None) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Expected bool config value.")
    return value


def _optional_str(value: TrainingConfigScalar | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_positive_int(value: TrainingConfigScalar | None) -> int | None:
    parsed = _optional_int(value)
    if parsed is None:
        return None
    if parsed < 1:
        raise ValueError("Expected positive int config value.")
    return parsed


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
    example_count: int,
    client_metrics: dict[str, float],
    update_id: str | None = None,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    payload_format: str = UpdatePayloadFormat.DIAGONAL_SCALE_UPDATE.value,
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
