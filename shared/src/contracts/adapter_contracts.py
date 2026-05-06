"""Shared adapter 상태/업데이트 payload와 직렬화 유틸리티.

이 모듈은 시스템/FL runtime에서 "서버와 agent 사이에 어떤 shared state/update를
주고받는가"를 정의하는 source of truth다.

현재 concrete family는 셋이다.

1. `diagonal_scale`
   - 임베딩 각 차원에 곱하는 scale 벡터를 공유하는 adapter family
2. `classifier_head`
   - 고정 임베딩 위 category별 선형 분류 head를 공유하는 family
3. `lora_classifier`
   - frozen backbone 위 LoRA adapter와 classifier head를 함께 공유하는 family

주의:

- 이 모듈은 현재 시스템/FL runtime 계약을 다룬다.
- 논문 트랙의 `central LoRA classifier` trainer가 이 payload를 그대로
  재사용해야 한다는 의미는 아니다.
- 다만 이후 시스템 FL translation에서 `lora_classifier` family를 열 때는 이 모듈이
  canonical state/update payload source of truth가 된다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from .common_types import TrainingScope
from .model_contracts import ModelManifestPayload

VECTOR_ADAPTER_STATE_V1 = "vector_adapter_state.v1"
VECTOR_ADAPTER_DELTA_V1 = "vector_adapter_delta.v1"
CLASSIFIER_HEAD_STATE_V1 = "classifier_head_state.v1"
CLASSIFIER_HEAD_DELTA_V1 = "classifier_head_delta.v1"
LORA_CLASSIFIER_STATE_V1 = "lora_classifier_state.v1"
LORA_CLASSIFIER_DELTA_V1 = "lora_classifier_delta.v1"
CURRENT_SHARED_ADAPTER_STATE_V1 = "current_shared_adapter_state.v1"
VectorAdapterStateSchemaVersion: TypeAlias = Literal["vector_adapter_state.v1"]
VectorAdapterDeltaSchemaVersion: TypeAlias = Literal["vector_adapter_delta.v1"]
ClassifierHeadStateSchemaVersion: TypeAlias = Literal["classifier_head_state.v1"]
ClassifierHeadDeltaSchemaVersion: TypeAlias = Literal["classifier_head_delta.v1"]
LoraClassifierStateSchemaVersion: TypeAlias = Literal["lora_classifier_state.v1"]
LoraClassifierDeltaSchemaVersion: TypeAlias = Literal["lora_classifier_delta.v1"]
CurrentSharedAdapterStateSchemaVersion: TypeAlias = Literal[
    "current_shared_adapter_state.v1"
]


class AdapterKind(StrEnum):
    """Shared adapter family discriminator."""

    DIAGONAL_SCALE = "diagonal_scale"
    CLASSIFIER_HEAD = "classifier_head"
    LORA_CLASSIFIER = "lora_classifier"


class SharedAdapterStatePayload(BaseModel):
    """전역 shared adapter 상태 payload 공통 필드.

    필드 의미:

    - `schema_version`: payload contract 버전
    - `adapter_kind`: 어떤 adapter family인지 식별하는 discriminator
    - `model_id`: 이 adapter가 붙는 backbone/model 식별자
    - `model_revision`: 서버가 현재 배포 중인 shared adapter revision
    - `training_scope`: `adapter_only` 같은 학습 범위 식별자
    - `updated_at`: 서버가 이 상태를 발행한 시각
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: VectorAdapterStateSchemaVersion = Field(
        default=VECTOR_ADAPTER_STATE_V1,
        description="Payload contract 버전.",
    )
    adapter_kind: str = Field(
        default=AdapterKind.DIAGONAL_SCALE.value,
        description=(
            "Adapter family discriminator. 현재 기본 구현은 diagonal_scale이며, "
            "다른 family는 registry 등록으로 확장한다."
        ),
    )
    model_id: str = Field(description="이 adapter가 결합되는 backbone/model 식별자.")
    model_revision: str = Field(description="서버가 발행한 shared adapter revision.")
    training_scope: TrainingScope = Field(
        description="이 상태가 적용되는 학습 범위 식별자."
    )
    updated_at: datetime = Field(description="서버가 이 상태를 기록한 UTC 시각.")


class DiagonalScaleAdapterStatePayload(SharedAdapterStatePayload):
    """현재 concrete 구현인 diagonal scale adapter 상태 payload.

    `dimension_scales[j]`는 임베딩 j번째 차원에 곱하는 전역 scale 값이다.
    runtime 적용식은 `x' = normalize(x ⊙ s)`다.
    """

    dimension_scales: list[float] = Field(
        description=(
            "임베딩 차원별 전역 scale 벡터. j번째 값은 j번째 차원을 얼마나 "
            "늘이거나 줄일지 나타낸다."
        )
    )

    @classmethod
    def identity(
        cls,
        *,
        model_id: str,
        model_revision: str,
        training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
        embedding_dim: int,
        updated_at: datetime,
        schema_version: str = VECTOR_ADAPTER_STATE_V1,
    ) -> "DiagonalScaleAdapterStatePayload":
        """아무 보정도 하지 않는 초기 adapter 상태를 만든다."""
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        return cls(
            schema_version=schema_version,
            adapter_kind=AdapterKind.DIAGONAL_SCALE.value,
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            updated_at=updated_at,
            dimension_scales=[1.0] * embedding_dim,
        )

    @property
    def embedding_dim(self) -> int:
        """현재 adapter가 기대하는 임베딩 차원을 반환한다."""
        return len(self.dimension_scales)

    def apply(self, embedding: Sequence[float]) -> list[float]:
        """임베딩에 차원별 scale을 적용하고 다시 L2 정규화한다."""
        if len(embedding) != self.embedding_dim:
            raise ValueError("Embedding dimension does not match adapter state.")
        scaled = [
            float(value) * float(scale)
            for value, scale in zip(embedding, self.dimension_scales, strict=True)
        ]
        norm = math.sqrt(sum(value * value for value in scaled))
        if norm == 0.0:
            raise ValueError("Adapter-transformed embedding norm must be non-zero.")
        return [value / norm for value in scaled]


class ClassifierHeadAdapterStatePayload(SharedAdapterStatePayload):
    """선형 classifier head family 상태 payload.

    이 family는 임베딩 자체를 변형하지 않고, 공통 임베딩 위에 category별
    linear head를 얹어 logits를 계산한다. `apply`는 prototype rebuild 등
    shared-state 공용 경로와의 호환을 위해 임베딩을 L2 정규화해 통과시킨다.
    """

    schema_version: ClassifierHeadStateSchemaVersion = Field(
        default=CLASSIFIER_HEAD_STATE_V1,
        description="Classifier head state payload contract 버전.",
    )
    label_weights: dict[str, list[float]] = Field(
        description="카테고리별 선형 head weight 벡터."
    )
    label_biases: dict[str, float] = Field(
        default_factory=dict,
        description="카테고리별 bias 항. 없는 카테고리는 0.0으로 간주한다.",
    )

    @model_validator(mode="after")
    def _validate_classifier_head_shape(self) -> "ClassifierHeadAdapterStatePayload":
        if not self.label_weights:
            raise ValueError("label_weights must not be empty.")

        labels = tuple(sorted(self.label_weights))
        dims = {len(weights) for weights in self.label_weights.values()}
        if dims == {0}:
            raise ValueError("Classifier head weights must be non-empty.")
        if len(dims) != 1:
            raise ValueError("All classifier head weight vectors must share one dim.")

        normalized_biases = {
            label: float(self.label_biases.get(label, 0.0)) for label in labels
        }
        extra_bias_labels = set(self.label_biases) - set(labels)
        if extra_bias_labels:
            raise ValueError(
                "Classifier head biases include unknown labels: "
                f"{sorted(extra_bias_labels)}"
            )
        self.label_biases = normalized_biases
        return self

    @classmethod
    def zero_initialized(
        cls,
        *,
        model_id: str,
        model_revision: str,
        labels: Sequence[str],
        embedding_dim: int,
        training_scope: TrainingScope = TrainingScope.HEAD_ONLY,
        updated_at: datetime,
    ) -> "ClassifierHeadAdapterStatePayload":
        """0으로 초기화된 선형 classifier head 상태를 만든다."""
        normalized_labels = tuple(
            sorted({str(label) for label in labels if str(label)})
        )
        if not normalized_labels:
            raise ValueError("labels must not be empty.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        return cls(
            schema_version=CLASSIFIER_HEAD_STATE_V1,
            adapter_kind=AdapterKind.CLASSIFIER_HEAD.value,
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            updated_at=updated_at,
            label_weights={label: [0.0] * embedding_dim for label in normalized_labels},
            label_biases={label: 0.0 for label in normalized_labels},
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.label_weights))

    @property
    def embedding_dim(self) -> int:
        first_label = self.labels[0]
        return len(self.label_weights[first_label])

    def apply(self, embedding: Sequence[float]) -> list[float]:
        if len(embedding) != self.embedding_dim:
            raise ValueError("Embedding dimension does not match classifier head.")
        norm = math.sqrt(sum(float(value) * float(value) for value in embedding))
        if norm == 0.0:
            raise ValueError("Classifier-head input embedding norm must be non-zero.")
        return [float(value) / norm for value in embedding]

    def compute_logits(self, embedding: Sequence[float]) -> dict[str, float]:
        """정규화된 임베딩에 대해 category별 linear logits를 계산한다."""
        if len(embedding) != self.embedding_dim:
            raise ValueError("Embedding dimension does not match classifier head.")
        return {
            label: sum(
                float(weight) * float(value)
                for weight, value in zip(weights, embedding, strict=True)
            )
            + float(self.label_biases.get(label, 0.0))
            for label, weights in sorted(self.label_weights.items())
        }


class LoraClassifierBackbonePayload(BaseModel):
    """LoRA-classifier가 결합되는 frozen text backbone/tokenizer 식별자."""

    model_config = ConfigDict(extra="forbid")

    backbone_model_id: str = Field(description="Frozen backbone model id.")
    backbone_revision: str = Field(description="Frozen backbone revision.")
    tokenizer_model_id: str = Field(description="Tokenizer model id.")
    tokenizer_revision: str = Field(description="Tokenizer revision.")
    pooling: str = Field(
        default="mean",
        description="Classifier head 입력 벡터를 만들 때 쓰는 pooling 방식.",
    )
    max_length: int = Field(
        gt=0,
        description="Tokenizer truncation/padding 기준 max_length.",
    )
    task_prefix: str = Field(
        default="",
        description="Embedding/query 입력 앞에 붙이는 task prefix.",
    )


class LoraClassifierConfigPayload(BaseModel):
    """LoRA/RSLoRA adapter config snapshot."""

    model_config = ConfigDict(extra="forbid")

    peft_adapter_name: str = Field(
        default="lora",
        description="PEFT adapter builder 이름. 예: lora, rslora.",
    )
    rank: int = Field(gt=0, description="LoRA rank.")
    alpha: int = Field(gt=0, description="LoRA scaling alpha.")
    dropout: float = Field(
        ge=0.0,
        le=1.0,
        description="LoRA dropout 비율.",
    )
    bias: str = Field(default="none", description="PEFT bias config.")
    target_modules: str | list[str] = Field(
        description="LoRA를 삽입할 backbone module selector."
    )
    use_rslora: bool = Field(
        default=False,
        description="RSLoRA scaling 사용 여부.",
    )

    @model_validator(mode="after")
    def _validate_target_modules(self) -> "LoraClassifierConfigPayload":
        if isinstance(self.target_modules, str):
            if not self.target_modules.strip():
                raise ValueError("target_modules must not be empty.")
            self.target_modules = self.target_modules.strip()
            return self
        normalized = tuple(
            str(value).strip() for value in self.target_modules if str(value).strip()
        )
        if not normalized:
            raise ValueError("target_modules must not be empty.")
        self.target_modules = list(normalized)
        return self


class LoraClassifierAdapterStatePayload(SharedAdapterStatePayload):
    """LoRA adapter와 classifier head를 함께 배포하는 shared family 상태.

    이 payload는 raw text나 client-local query state를 포함하지 않는다. 서버가
    배포할 shared artifact의 식별자와 scaffold metadata만 담는다.
    """

    schema_version: LoraClassifierStateSchemaVersion = Field(
        default=LORA_CLASSIFIER_STATE_V1,
        description="LoRA-classifier state payload contract 버전.",
    )
    adapter_kind: str = Field(
        default=AdapterKind.LORA_CLASSIFIER.value,
        description="Adapter family discriminator.",
    )
    backbone: LoraClassifierBackbonePayload = Field(
        description="Frozen backbone/tokenizer snapshot."
    )
    lora_config: LoraClassifierConfigPayload = Field(
        description="LoRA/RSLoRA adapter config snapshot."
    )
    label_schema: list[str] = Field(description="Classifier head label 순서.")
    lora_adapter_artifact_ref: str | None = Field(
        default=None,
        description="서버가 소유한 LoRA adapter artifact ref.",
    )
    classifier_head_artifact_ref: str | None = Field(
        default=None,
        description="서버가 소유한 classifier head artifact ref.",
    )
    artifact_format: str = Field(
        default="artifact_ref",
        description="LoRA/classifier weight 저장 방식 식별자.",
    )

    @model_validator(mode="after")
    def _validate_lora_classifier_state(
        self,
    ) -> "LoraClassifierAdapterStatePayload":
        self.label_schema = _normalize_label_schema(self.label_schema)
        if not self.artifact_format.strip():
            raise ValueError("artifact_format must not be empty.")
        self.artifact_format = self.artifact_format.strip()
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label_schema)


class SharedAdapterUpdatePayload(BaseModel):
    """로컬 학습이 생성한 shared adapter update payload 공통 필드.

    필드 의미:

    - `schema_version`: payload contract 버전
    - `adapter_kind`: 어떤 adapter family의 update인지 식별자
    - `model_id`: 이 update가 대상으로 삼는 backbone/model 식별자
    - `base_model_revision`: 어떤 전역 revision을 기준으로 계산한 update인지
    - `training_scope`: `adapter_only` 같은 학습 범위 식별자
    - `example_count`: update 계산에 실제 반영된 로컬 예시 수
    - `created_at`: agent가 update를 기록한 시각
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: VectorAdapterDeltaSchemaVersion = Field(
        default=VECTOR_ADAPTER_DELTA_V1,
        description="Payload contract 버전.",
    )
    adapter_kind: str = Field(
        default=AdapterKind.DIAGONAL_SCALE.value,
        description=(
            "Adapter family discriminator. 현재 기본 구현은 diagonal_scale이며, "
            "다른 family는 registry 등록으로 확장한다."
        ),
    )
    model_id: str = Field(
        description="이 update가 대상으로 삼는 backbone/model 식별자."
    )
    base_model_revision: str = Field(
        description="로컬 update가 계산된 기준 shared adapter revision."
    )
    training_scope: TrainingScope = Field(
        description="이 update가 속한 학습 범위 식별자."
    )
    example_count: int = Field(
        ge=0,
        description="실제 update 계산에 반영된 로컬 예시 수.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Agent가 이 update payload를 생성한 UTC 시각.",
    )


class DiagonalScaleAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """현재 concrete 구현인 diagonal scale adapter update payload.

    `dimension_deltas[j]`는 j번째 차원 scale에 더할 변화량이다.
    현재 heuristic 구현에서는 accepted example 임베딩 평균 방향에서 유도된
    전역 차원 보정량이며, 특정 prototype 좌표로 직접 이동시키는 값은 아니다.
    """

    dimension_deltas: list[float] = Field(description="차원별 scale에 더할 delta 벡터.")
    mean_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Update에 반영된 accepted example들의 평균 confidence.",
    )
    mean_margin: float | None = Field(
        default=None,
        description="Accepted example들의 평균 top1-top2 margin.",
    )
    label_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Accepted example의 pseudo-label 분포. drift 관찰용 메타데이터다.",
    )

    @property
    def embedding_dim(self) -> int:
        """delta가 적용되는 임베딩 차원을 반환한다."""
        return len(self.dimension_deltas)

    def l2_norm(self) -> float:
        """delta 벡터의 L2 norm을 반환한다."""
        return math.sqrt(sum(value * value for value in self.dimension_deltas))


class ClassifierHeadAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """선형 classifier head family update payload."""

    schema_version: ClassifierHeadDeltaSchemaVersion = Field(
        default=CLASSIFIER_HEAD_DELTA_V1,
        description="Classifier head update payload contract 버전.",
    )
    label_weight_deltas: dict[str, list[float]] = Field(
        description="카테고리별 weight delta 벡터."
    )
    label_bias_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="카테고리별 bias delta. 없는 카테고리는 0.0으로 간주한다.",
    )
    mean_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Update에 반영된 accepted example들의 평균 confidence.",
    )
    mean_margin: float | None = Field(
        default=None,
        description="Accepted example들의 평균 top1-top2 margin.",
    )
    label_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Accepted example의 pseudo-label 분포. drift 관찰용 메타데이터다.",
    )

    @model_validator(mode="after")
    def _validate_classifier_head_delta_shape(
        self,
    ) -> "ClassifierHeadAdapterUpdatePayload":
        if not self.label_weight_deltas:
            raise ValueError("label_weight_deltas must not be empty.")

        labels = tuple(sorted(self.label_weight_deltas))
        dims = {len(weights) for weights in self.label_weight_deltas.values()}
        if dims == {0}:
            raise ValueError("Classifier head delta vectors must be non-empty.")
        if len(dims) != 1:
            raise ValueError("All classifier head delta vectors must share one dim.")

        normalized_biases = {
            label: float(self.label_bias_deltas.get(label, 0.0)) for label in labels
        }
        extra_bias_labels = set(self.label_bias_deltas) - set(labels)
        if extra_bias_labels:
            raise ValueError(
                "Classifier head bias deltas include unknown labels: "
                f"{sorted(extra_bias_labels)}"
            )
        self.label_bias_deltas = normalized_biases
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.label_weight_deltas))

    @property
    def embedding_dim(self) -> int:
        first_label = self.labels[0]
        return len(self.label_weight_deltas[first_label])

    def l2_norm(self) -> float:
        """weight/bias delta 전체의 L2 norm을 반환한다."""
        squared_weight_norm = sum(
            float(value) * float(value)
            for deltas in self.label_weight_deltas.values()
            for value in deltas
        )
        squared_bias_norm = sum(
            float(value) * float(value) for value in self.label_bias_deltas.values()
        )
        return math.sqrt(squared_weight_norm + squared_bias_norm)


class LoraClassifierAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """LoRA-classifier family update payload.

    LoRA weight delta는 커질 수 있으므로 artifact ref 기반을 canonical 경로로
    열어 둔다. 작은 smoke나 deterministic unit 검증은 optional inline delta를
    쓸 수 있지만, runtime은 둘 중 어떤 표현이 들어왔는지 명시적으로 확인해야 한다.
    """

    schema_version: LoraClassifierDeltaSchemaVersion = Field(
        default=LORA_CLASSIFIER_DELTA_V1,
        description="LoRA-classifier update payload contract 버전.",
    )
    adapter_kind: str = Field(
        default=AdapterKind.LORA_CLASSIFIER.value,
        description="Adapter family discriminator.",
    )
    backbone: LoraClassifierBackbonePayload = Field(
        description="Frozen backbone/tokenizer snapshot."
    )
    lora_config: LoraClassifierConfigPayload = Field(
        description="LoRA/RSLoRA adapter config snapshot."
    )
    label_schema: list[str] = Field(description="Classifier head label 순서.")
    lora_delta_artifact_ref: str | None = Field(
        default=None,
        description="서버가 해석할 LoRA adapter delta artifact ref.",
    )
    classifier_head_delta_artifact_ref: str | None = Field(
        default=None,
        description="서버가 해석할 classifier head delta artifact ref.",
    )
    lora_parameter_deltas: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "선택적 inline LoRA parameter delta. 큰 update의 기본 경로가 아니다."
        ),
    )
    classifier_head_weight_deltas: dict[str, list[float]] | None = Field(
        default=None,
        description="선택적 inline classifier head weight delta.",
    )
    classifier_head_bias_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="선택적 inline classifier head bias delta.",
    )
    delta_format: str = Field(
        default="artifact_ref",
        description="Update weight 표현 방식 식별자.",
    )
    mean_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Update에 반영된 accepted example들의 평균 confidence.",
    )
    mean_margin: float | None = Field(
        default=None,
        description="Accepted example들의 평균 top1-top2 margin.",
    )
    label_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Accepted example의 pseudo-label 분포. drift 관찰용 메타데이터다.",
    )
    delta_l2_norm: float | None = Field(
        default=None,
        ge=0.0,
        description="Artifact 기반 update가 보고한 전체 delta L2 norm.",
    )

    @model_validator(mode="after")
    def _validate_lora_classifier_delta(
        self,
    ) -> "LoraClassifierAdapterUpdatePayload":
        self.label_schema = _normalize_label_schema(self.label_schema)
        self.delta_format = self.delta_format.strip()
        if not self.delta_format:
            raise ValueError("delta_format must not be empty.")
        if not self._has_update_material():
            raise ValueError(
                "LoRA-classifier update requires artifact refs or inline deltas."
            )
        if self.classifier_head_weight_deltas is not None:
            _validate_label_vector_mapping(
                self.classifier_head_weight_deltas,
                labels=tuple(self.label_schema),
                field_name="classifier_head_weight_deltas",
            )
            self.classifier_head_bias_deltas = _normalize_label_scalar_mapping(
                self.classifier_head_bias_deltas,
                labels=tuple(self.label_schema),
                field_name="classifier_head_bias_deltas",
            )
        if self.lora_parameter_deltas is not None:
            _validate_non_empty_vector_mapping(
                self.lora_parameter_deltas,
                field_name="lora_parameter_deltas",
            )
        return self

    def _has_update_material(self) -> bool:
        return any(
            (
                self.lora_delta_artifact_ref,
                self.classifier_head_delta_artifact_ref,
                self.lora_parameter_deltas,
                self.classifier_head_weight_deltas,
                self.classifier_head_bias_deltas,
            )
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label_schema)

    def l2_norm(self) -> float:
        """Inline delta가 있으면 계산하고, artifact 기반이면 reported norm을 쓴다."""
        if self.delta_l2_norm is not None:
            return self.delta_l2_norm
        squared_norm = 0.0
        if self.lora_parameter_deltas is not None:
            squared_norm += _squared_vector_mapping_norm(self.lora_parameter_deltas)
        if self.classifier_head_weight_deltas is not None:
            squared_norm += _squared_vector_mapping_norm(
                self.classifier_head_weight_deltas
            )
        squared_norm += sum(
            float(value) * float(value)
            for value in self.classifier_head_bias_deltas.values()
        )
        return math.sqrt(squared_norm)


class CurrentSharedAdapterStatePayload(BaseModel):
    """서버 current manifest와 실제 shared adapter state를 함께 내려주는 payload.

    `ModelManifest.artifact_ref`는 서버 내부 참조일 수 있으므로 agent가 직접
    해석하지 않는다. Agent는 이 payload의 `state`를 로컬 캐시에 저장하고
    `manifest`를 같은 revision의 메타데이터로 함께 보관한다.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: CurrentSharedAdapterStateSchemaVersion = Field(
        default=CURRENT_SHARED_ADAPTER_STATE_V1,
        description="Current shared adapter state 응답 contract 버전.",
    )
    manifest: ModelManifestPayload = Field(
        description="서버가 current로 활성화한 model manifest."
    )
    state: SerializeAsAny[SharedAdapterStatePayload] = Field(
        description="manifest revision이 가리키는 실제 shared adapter state."
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_state_payload(cls, source: object) -> object:
        if not isinstance(source, Mapping):
            return source
        data = dict(source)
        raw_state = data.get("state")
        if isinstance(raw_state, Mapping):
            data["state"] = parse_shared_adapter_state_payload(raw_state)
        return data

    @model_validator(mode="after")
    def _validate_manifest_state_alignment(
        self,
    ) -> "CurrentSharedAdapterStatePayload":
        if self.manifest.model_id != self.state.model_id:
            raise ValueError("Current shared state model_id must match manifest.")
        if self.manifest.model_revision != self.state.model_revision:
            raise ValueError("Current shared state model_revision must match manifest.")
        if self.manifest.training_scope != self.state.training_scope:
            raise ValueError("Current shared state training_scope must match manifest.")
        return self


def _normalize_label_schema(labels: Sequence[str]) -> list[str]:
    normalized = [str(label).strip() for label in labels if str(label).strip()]
    if not normalized:
        raise ValueError("label_schema must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("label_schema must not contain duplicates.")
    return normalized


def _validate_non_empty_vector_mapping(
    values: Mapping[str, Sequence[float]],
    *,
    field_name: str,
) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    for key, vector in values.items():
        if not str(key).strip():
            raise ValueError(f"{field_name} keys must not be empty.")
        if not vector:
            raise ValueError(f"{field_name} vectors must not be empty.")


def _validate_label_vector_mapping(
    values: Mapping[str, Sequence[float]],
    *,
    labels: Sequence[str],
    field_name: str,
) -> None:
    _validate_non_empty_vector_mapping(values, field_name=field_name)
    if set(values) != set(labels):
        raise ValueError(f"{field_name} keys must match label_schema.")
    dims = {len(vector) for vector in values.values()}
    if len(dims) != 1:
        raise ValueError(f"{field_name} vectors must share one dimension.")


def _normalize_label_scalar_mapping(
    values: Mapping[str, float],
    *,
    labels: Sequence[str],
    field_name: str,
) -> dict[str, float]:
    extra_labels = set(values) - set(labels)
    if extra_labels:
        raise ValueError(
            f"{field_name} includes unknown labels: {sorted(extra_labels)}"
        )
    return {label: float(values.get(label, 0.0)) for label in labels}


def _squared_vector_mapping_norm(values: Mapping[str, Sequence[float]]) -> float:
    return sum(
        float(value) * float(value) for vector in values.values() for value in vector
    )


VectorAdapterStatePayload = DiagonalScaleAdapterStatePayload
VectorAdapterDeltaPayload = DiagonalScaleAdapterUpdatePayload
VectorAdapterState = VectorAdapterStatePayload
VectorAdapterDelta = VectorAdapterDeltaPayload
ClassifierHeadStatePayload = ClassifierHeadAdapterStatePayload
ClassifierHeadDeltaPayload = ClassifierHeadAdapterUpdatePayload
ClassifierHeadState = ClassifierHeadStatePayload
ClassifierHeadDelta = ClassifierHeadDeltaPayload
LoraClassifierStatePayload = LoraClassifierAdapterStatePayload
LoraClassifierDeltaPayload = LoraClassifierAdapterUpdatePayload
LoraClassifierState = LoraClassifierStatePayload
LoraClassifierDelta = LoraClassifierDeltaPayload

_STATE_PAYLOAD_TYPES: dict[str, type[SharedAdapterStatePayload]] = {}
_UPDATE_PAYLOAD_TYPES: dict[str, type[SharedAdapterUpdatePayload]] = {}


def register_shared_adapter_state_payload_type(
    adapter_kind: str,
    payload_type: type[SharedAdapterStatePayload],
) -> None:
    """adapter family별 state payload 타입을 registry에 등록한다."""

    _STATE_PAYLOAD_TYPES[adapter_kind.strip().lower()] = payload_type


def register_shared_adapter_update_payload_type(
    adapter_kind: str,
    payload_type: type[SharedAdapterUpdatePayload],
) -> None:
    """adapter family별 update payload 타입을 registry에 등록한다."""

    _UPDATE_PAYLOAD_TYPES[adapter_kind.strip().lower()] = payload_type


def register_shared_adapter_payload_family(
    adapter_kind: str,
    *,
    state_payload_type: type[SharedAdapterStatePayload],
    update_payload_type: type[SharedAdapterUpdatePayload],
) -> None:
    """adapter family의 state/update payload 타입을 함께 등록한다."""

    register_shared_adapter_state_payload_type(adapter_kind, state_payload_type)
    register_shared_adapter_update_payload_type(adapter_kind, update_payload_type)


def _dump_payload(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_payload_data(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_shared_adapter_state_payload(path: Path) -> SharedAdapterStatePayload:
    """JSON 파일에서 shared adapter state payload를 읽는다."""
    return parse_shared_adapter_state_payload(_load_payload_data(path))


def parse_shared_adapter_state_payload(
    source: Mapping[str, object] | SharedAdapterStatePayload,
) -> SharedAdapterStatePayload:
    """mapping 또는 payload instance를 registered state payload로 정규화한다."""
    if isinstance(source, SharedAdapterStatePayload):
        return source
    data = dict(source)
    adapter_kind = (
        str(data.get("adapter_kind", AdapterKind.DIAGONAL_SCALE.value)).strip().lower()
    )
    payload_type = _STATE_PAYLOAD_TYPES.get(adapter_kind)
    if payload_type is None:
        raise ValueError(f"Unsupported shared adapter state kind: {adapter_kind}")
    return payload_type.model_validate(data)


def dump_shared_adapter_state_payload(
    path: Path,
    payload: SharedAdapterStatePayload,
) -> None:
    """shared adapter state payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_shared_adapter_update_payload(path: Path) -> SharedAdapterUpdatePayload:
    """JSON 파일에서 shared adapter update payload를 읽는다."""
    return parse_shared_adapter_update_payload(_load_payload_data(path))


def parse_shared_adapter_update_payload(
    source: Mapping[str, object] | SharedAdapterUpdatePayload,
) -> SharedAdapterUpdatePayload:
    """mapping 또는 payload instance를 registered update payload로 정규화한다."""
    if isinstance(source, SharedAdapterUpdatePayload):
        return source
    data = dict(source)
    adapter_kind = (
        str(data.get("adapter_kind", AdapterKind.DIAGONAL_SCALE.value)).strip().lower()
    )
    payload_type = _UPDATE_PAYLOAD_TYPES.get(adapter_kind)
    if payload_type is None:
        raise ValueError(f"Unsupported shared adapter update kind: {adapter_kind}")
    return payload_type.model_validate(data)


def dump_shared_adapter_update_payload(
    path: Path,
    payload: SharedAdapterUpdatePayload,
) -> None:
    """shared adapter update payload를 JSON 파일로 기록한다."""
    _dump_payload(path, payload)


def load_vector_adapter_state_payload(path: Path) -> VectorAdapterStatePayload:
    """JSON 파일에서 diagonal scale adapter state payload를 읽는다."""
    payload = load_shared_adapter_state_payload(path)
    if not isinstance(payload, DiagonalScaleAdapterStatePayload):
        raise ValueError(
            "Expected diagonal_scale adapter state payload, "
            f"got {payload.adapter_kind}."
        )
    return payload


def dump_vector_adapter_state_payload(
    path: Path,
    payload: VectorAdapterStatePayload,
) -> None:
    """diagonal scale adapter state payload를 JSON 파일로 기록한다."""
    dump_shared_adapter_state_payload(path, payload)


def load_vector_adapter_delta_payload(path: Path) -> VectorAdapterDeltaPayload:
    """JSON 파일에서 diagonal scale adapter update payload를 읽는다."""
    payload = load_shared_adapter_update_payload(path)
    if not isinstance(payload, DiagonalScaleAdapterUpdatePayload):
        raise ValueError(
            "Expected diagonal_scale adapter update payload, "
            f"got {payload.adapter_kind}."
        )
    return payload


def dump_vector_adapter_delta_payload(
    path: Path,
    payload: VectorAdapterDeltaPayload,
) -> None:
    """diagonal scale adapter update payload를 JSON 파일로 기록한다."""
    dump_shared_adapter_update_payload(path, payload)


# ------------------------------------------------------------------ #
# Factory 함수                                                         #
# ------------------------------------------------------------------ #


def make_identity_state_payload(
    *,
    model_id: str,
    model_revision: str,
    embedding_dim: int,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    updated_at: datetime | None = None,
) -> DiagonalScaleAdapterStatePayload:
    """모든 차원 scale=1.0 인 identity(단위) adapter state payload를 만든다.

    새 round 시작 또는 테스트 fixture 생성에 사용한다.

    >>> state = make_identity_state_payload(
    ...     model_id="bg-m3", model_revision="rev_001", embedding_dim=768
    ... )
    """
    return DiagonalScaleAdapterStatePayload.identity(
        model_id=model_id,
        model_revision=model_revision,
        training_scope=TrainingScope(training_scope),
        embedding_dim=embedding_dim,
        updated_at=updated_at or datetime.now(tz=timezone.utc),
    )


def make_diagonal_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    dimension_deltas: list[float],
    example_count: int,
    mean_confidence: float,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    created_at: datetime | None = None,
) -> DiagonalScaleAdapterUpdatePayload:
    """diagonal scale adapter update payload를 만드는 표준 factory.

    >>> delta = make_diagonal_delta_payload(
    ...     model_id="bg-m3",
    ...     base_model_revision="rev_001",
    ...     dimension_deltas=[0.01] * 768,
    ...     example_count=10,
    ...     mean_confidence=0.85,
    ... )
    """
    return DiagonalScaleAdapterUpdatePayload(
        schema_version=VECTOR_ADAPTER_DELTA_V1,
        adapter_kind=AdapterKind.DIAGONAL_SCALE.value,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        dimension_deltas=dimension_deltas,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
    )


def make_zero_classifier_head_state_payload(
    *,
    model_id: str,
    model_revision: str,
    labels: Sequence[str],
    embedding_dim: int,
    training_scope: TrainingScope = TrainingScope.HEAD_ONLY,
    updated_at: datetime | None = None,
) -> ClassifierHeadAdapterStatePayload:
    """0으로 초기화된 classifier head state payload를 만든다."""
    return ClassifierHeadAdapterStatePayload.zero_initialized(
        model_id=model_id,
        model_revision=model_revision,
        labels=labels,
        embedding_dim=embedding_dim,
        training_scope=TrainingScope(training_scope),
        updated_at=updated_at or datetime.now(tz=timezone.utc),
    )


def make_classifier_head_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    label_weight_deltas: dict[str, list[float]],
    example_count: int,
    mean_confidence: float,
    training_scope: TrainingScope = TrainingScope.HEAD_ONLY,
    label_bias_deltas: dict[str, float] | None = None,
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    created_at: datetime | None = None,
) -> ClassifierHeadAdapterUpdatePayload:
    """classifier head update payload를 만드는 표준 factory."""
    return ClassifierHeadAdapterUpdatePayload(
        schema_version=CLASSIFIER_HEAD_DELTA_V1,
        adapter_kind=AdapterKind.CLASSIFIER_HEAD.value,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        label_weight_deltas=label_weight_deltas,
        label_bias_deltas=label_bias_deltas or {},
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
    )


def make_lora_classifier_state_payload(
    *,
    model_id: str,
    model_revision: str,
    backbone: LoraClassifierBackbonePayload | Mapping[str, object],
    lora_config: LoraClassifierConfigPayload | Mapping[str, object],
    label_schema: Sequence[str],
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    lora_adapter_artifact_ref: str | None = None,
    classifier_head_artifact_ref: str | None = None,
    artifact_format: str = "artifact_ref",
    updated_at: datetime | None = None,
) -> LoraClassifierAdapterStatePayload:
    """LoRA-classifier state payload를 만드는 표준 factory."""

    return LoraClassifierAdapterStatePayload(
        schema_version=LORA_CLASSIFIER_STATE_V1,
        adapter_kind=AdapterKind.LORA_CLASSIFIER.value,
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at or datetime.now(tz=timezone.utc),
        backbone=backbone,
        lora_config=lora_config,
        label_schema=list(label_schema),
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_format,
    )


def make_lora_classifier_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    backbone: LoraClassifierBackbonePayload | Mapping[str, object],
    lora_config: LoraClassifierConfigPayload | Mapping[str, object],
    label_schema: Sequence[str],
    example_count: int,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    lora_delta_artifact_ref: str | None = None,
    classifier_head_delta_artifact_ref: str | None = None,
    lora_parameter_deltas: dict[str, list[float]] | None = None,
    classifier_head_weight_deltas: dict[str, list[float]] | None = None,
    classifier_head_bias_deltas: dict[str, float] | None = None,
    delta_format: str = "artifact_ref",
    mean_confidence: float | None = None,
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    delta_l2_norm: float | None = None,
    created_at: datetime | None = None,
) -> LoraClassifierAdapterUpdatePayload:
    """LoRA-classifier update payload를 만드는 표준 factory."""

    return LoraClassifierAdapterUpdatePayload(
        schema_version=LORA_CLASSIFIER_DELTA_V1,
        adapter_kind=AdapterKind.LORA_CLASSIFIER.value,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        backbone=backbone,
        lora_config=lora_config,
        label_schema=list(label_schema),
        lora_delta_artifact_ref=lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=classifier_head_delta_artifact_ref,
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas or {},
        delta_format=delta_format,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
        delta_l2_norm=delta_l2_norm,
    )


def make_current_shared_adapter_state_payload(
    *,
    manifest: ModelManifestPayload,
    state: SharedAdapterStatePayload,
) -> CurrentSharedAdapterStatePayload:
    """서버 current manifest/state pair payload를 만든다."""
    return CurrentSharedAdapterStatePayload(
        schema_version=CURRENT_SHARED_ADAPTER_STATE_V1,
        manifest=manifest,
        state=state,
    )


register_shared_adapter_payload_family(
    AdapterKind.DIAGONAL_SCALE.value,
    state_payload_type=DiagonalScaleAdapterStatePayload,
    update_payload_type=DiagonalScaleAdapterUpdatePayload,
)
register_shared_adapter_payload_family(
    AdapterKind.CLASSIFIER_HEAD.value,
    state_payload_type=ClassifierHeadAdapterStatePayload,
    update_payload_type=ClassifierHeadAdapterUpdatePayload,
)
register_shared_adapter_payload_family(
    AdapterKind.LORA_CLASSIFIER.value,
    state_payload_type=LoraClassifierAdapterStatePayload,
    update_payload_type=LoraClassifierAdapterUpdatePayload,
)
