"""Shared adapter base payload contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from shared.src.contracts.common_types import TrainingScope
from shared.src.contracts.model_contracts import ModelManifestPayload

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
    """전역 shared adapter 상태 payload 공통 필드."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(description="Payload contract 버전.")
    adapter_kind: str = Field(
        description="Adapter family discriminator.",
    )
    model_id: str = Field(description="이 adapter가 결합되는 backbone/model 식별자.")
    model_revision: str = Field(description="서버가 발행한 shared adapter revision.")
    training_scope: TrainingScope = Field(
        description="이 상태가 적용되는 학습 범위 식별자."
    )
    updated_at: datetime = Field(description="서버가 이 상태를 기록한 UTC 시각.")


class SharedAdapterUpdatePayload(BaseModel):
    """로컬 학습이 생성한 shared adapter update payload 공통 필드."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(description="Payload contract 버전.")
    adapter_kind: str = Field(
        description="Adapter family discriminator.",
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


class CurrentSharedAdapterStatePayload(BaseModel):
    """서버 current manifest와 실제 shared adapter state를 함께 내려주는 payload."""

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
            from shared.src.contracts.adapter_contract_families.registry import (
                parse_shared_adapter_state_payload,
            )

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


def normalize_label_schema(labels) -> list[str]:
    normalized = [str(label).strip() for label in labels if str(label).strip()]
    if not normalized:
        raise ValueError("label_schema must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("label_schema must not contain duplicates.")
    return normalized


def validate_non_empty_vector_mapping(
    values: Mapping[str, object],
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


def validate_label_vector_mapping(
    values: Mapping[str, object],
    *,
    labels,
    field_name: str,
) -> None:
    validate_non_empty_vector_mapping(values, field_name=field_name)
    if set(values) != set(labels):
        raise ValueError(f"{field_name} keys must match label_schema.")
    dims = {len(vector) for vector in values.values()}
    if len(dims) != 1:
        raise ValueError(f"{field_name} vectors must share one dimension.")


def normalize_label_scalar_mapping(
    values: Mapping[str, float],
    *,
    labels,
    field_name: str,
) -> dict[str, float]:
    extra_labels = set(values) - set(labels)
    if extra_labels:
        raise ValueError(
            f"{field_name} includes unknown labels: {sorted(extra_labels)}"
        )
    return {label: float(values.get(label, 0.0)) for label in labels}


def squared_vector_mapping_norm(values: Mapping[str, object]) -> float:
    return sum(
        float(value) * float(value) for vector in values.values() for value in vector
    )
