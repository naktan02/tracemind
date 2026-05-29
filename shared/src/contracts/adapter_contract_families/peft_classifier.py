"""PEFT-classifier shared adapter contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import (
    PEFT_CLASSIFIER_DELTA_V2,
    PEFT_CLASSIFIER_STATE_V2,
    AdapterKind,
    PeftClassifierDeltaSchemaVersion,
    PeftClassifierStateSchemaVersion,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
    normalize_label_scalar_mapping,
    normalize_label_schema,
    squared_vector_mapping_norm,
    validate_label_vector_mapping,
    validate_non_empty_vector_mapping,
)

PeftAdapterConfigValue: TypeAlias = (
    str | int | float | bool | list[str] | list[int] | list[float]
)

PEFT_CLASSIFIER_ADAPTER_KIND = AdapterKind.PEFT_CLASSIFIER.value
PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT = "peft_classifier_update"
PEFT_CLASSIFIER_ACCEPTED_UPDATE_PAYLOAD_FORMATS = (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
)


class PeftClassifierBackbonePayload(BaseModel):
    """PEFT-classifier가 결합되는 frozen text backbone/tokenizer 식별자."""

    model_config = ConfigDict(extra="forbid")

    backbone_model_id: str = Field(description="Frozen backbone model id.")
    backbone_revision: str = Field(description="Frozen backbone revision.")
    tokenizer_model_id: str = Field(description="Tokenizer model id.")
    tokenizer_revision: str = Field(description="Tokenizer revision.")
    pooling: str = Field(default="mean")
    max_length: int = Field(gt=0)
    task_prefix: str = Field(default="")


class PeftAdapterConfigPayload(BaseModel):
    """LoRA/DoRA처럼 교체 가능한 PEFT adapter mechanism 설정 snapshot."""

    model_config = ConfigDict(extra="forbid")

    peft_adapter_name: str = Field(description="예: lora, dora.")
    parameters: dict[str, PeftAdapterConfigValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_peft_adapter_config(self) -> "PeftAdapterConfigPayload":
        adapter_name = self.peft_adapter_name.strip().lower()
        if not adapter_name:
            raise ValueError("peft_adapter_name must not be empty.")
        self.peft_adapter_name = adapter_name

        normalized: dict[str, PeftAdapterConfigValue] = {}
        for raw_key, value in self.parameters.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(
                    "peft_adapter_config.parameters keys must not be empty."
                )
            normalized[key] = value
        self.parameters = normalized
        return self


class PeftClassifierAdapterStatePayload(SharedAdapterStatePayload):
    """PEFT adapter와 classifier head를 함께 배포하는 shared family state."""

    schema_version: PeftClassifierStateSchemaVersion = Field(
        default=PEFT_CLASSIFIER_STATE_V2,
        description="PEFT-classifier state payload contract 버전.",
    )
    adapter_kind: str = Field(default=PEFT_CLASSIFIER_ADAPTER_KIND)
    backbone: PeftClassifierBackbonePayload
    peft_adapter_config: PeftAdapterConfigPayload
    label_schema: list[str]
    peft_adapter_artifact_ref: str | None = Field(
        default=None,
        description=(
            "Server-published PEFT adapter state artifact. LoRA/DoRA 같은 "
            "mechanism은 peft_adapter_config.peft_adapter_name이 식별한다."
        ),
    )
    classifier_head_artifact_ref: str | None = Field(
        default=None,
        description="Server-published classifier-head state artifact.",
    )
    artifact_format: str = "artifact_ref"

    @model_validator(mode="after")
    def _validate_peft_classifier_state(self) -> "PeftClassifierAdapterStatePayload":
        self.label_schema = normalize_label_schema(self.label_schema)
        if not self.artifact_format.strip():
            raise ValueError("artifact_format must not be empty.")
        self.artifact_format = self.artifact_format.strip()
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label_schema)

    @property
    def peft_adapter_name(self) -> str:
        return self.peft_adapter_config.peft_adapter_name

    def apply(self, embedding: Sequence[float]) -> list[float]:
        vector = [float(value) for value in embedding]
        if not vector:
            raise ValueError("PEFT-classifier input embedding must not be empty.")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("PEFT-classifier input embedding norm must be non-zero.")
        return [value / norm for value in vector]


class PeftClassifierPartitionDeltaPayload(BaseModel):
    """하나의 logical partition이 가진 PEFT/head delta payload."""

    model_config = ConfigDict(extra="forbid")

    peft_parameter_deltas: dict[str, list[float]] = Field(default_factory=dict)
    classifier_head_weight_deltas: dict[str, list[float]] = Field(default_factory=dict)
    classifier_head_bias_deltas: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_partition_delta(self) -> "PeftClassifierPartitionDeltaPayload":
        if self.peft_parameter_deltas:
            validate_non_empty_vector_mapping(
                self.peft_parameter_deltas,
                field_name="partitioned_deltas.peft_parameter_deltas",
            )
        if self.classifier_head_weight_deltas:
            validate_non_empty_vector_mapping(
                self.classifier_head_weight_deltas,
                field_name="partitioned_deltas.classifier_head_weight_deltas",
            )
        if not self._has_update_material():
            raise ValueError("partitioned_deltas partitions must not be empty.")
        normalized_biases: dict[str, float] = {}
        for label, value in self.classifier_head_bias_deltas.items():
            normalized_label = str(label).strip()
            if not normalized_label:
                raise ValueError(
                    "partitioned_deltas.classifier_head_bias_deltas keys must not "
                    "be empty."
                )
            normalized_biases[normalized_label] = float(value)
        self.classifier_head_bias_deltas = normalized_biases
        return self

    def _has_update_material(self) -> bool:
        return any(
            (
                self.peft_parameter_deltas,
                self.classifier_head_weight_deltas,
                self.classifier_head_bias_deltas,
            )
        )

    def squared_l2_norm(self) -> float:
        """partition 안 delta의 squared L2 norm을 계산한다."""

        squared_norm = 0.0
        if self.peft_parameter_deltas:
            squared_norm += squared_vector_mapping_norm(self.peft_parameter_deltas)
        if self.classifier_head_weight_deltas:
            squared_norm += squared_vector_mapping_norm(
                self.classifier_head_weight_deltas
            )
        squared_norm += sum(
            float(value) * float(value)
            for value in self.classifier_head_bias_deltas.values()
        )
        return squared_norm


class PeftClassifierAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """PEFT-classifier family update payload."""

    schema_version: PeftClassifierDeltaSchemaVersion = Field(
        default=PEFT_CLASSIFIER_DELTA_V2,
        description="PEFT-classifier update payload contract 버전.",
    )
    adapter_kind: str = Field(default=PEFT_CLASSIFIER_ADAPTER_KIND)
    canonical_update_payload_format: ClassVar[str] = (
        PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
    )
    accepted_update_payload_formats: ClassVar[tuple[str, ...]] = (
        PEFT_CLASSIFIER_ACCEPTED_UPDATE_PAYLOAD_FORMATS
    )
    backbone: PeftClassifierBackbonePayload
    peft_adapter_config: PeftAdapterConfigPayload
    label_schema: list[str]
    peft_adapter_delta_artifact_ref: str | None = None
    classifier_head_delta_artifact_ref: str | None = None
    peft_parameter_deltas: dict[str, list[float]] | None = None
    classifier_head_weight_deltas: dict[str, list[float]] | None = None
    classifier_head_bias_deltas: dict[str, float] = Field(default_factory=dict)
    partitioned_deltas: dict[str, PeftClassifierPartitionDeltaPayload] | None = None
    partitioned_deltas_artifact_ref: str | None = Field(
        default=None,
        description="Server-owned artifact ref containing partitioned_deltas.",
    )
    delta_format: str = "artifact_ref"
    mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_margin: float | None = None
    label_counts: dict[str, int] = Field(default_factory=dict)
    delta_l2_norm: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_partitioned_delta_values(cls, source: object) -> object:
        if not isinstance(source, Mapping):
            return source
        data = dict(source)
        raw_partitioned_deltas = data.get("partitioned_deltas")
        if isinstance(raw_partitioned_deltas, Mapping):
            data["partitioned_deltas"] = {
                str(name): PeftClassifierPartitionDeltaPayload.model_validate(partition)
                for name, partition in raw_partitioned_deltas.items()
            }
        return data

    @model_validator(mode="after")
    def _validate_peft_classifier_delta(self) -> "PeftClassifierAdapterUpdatePayload":
        self.label_schema = normalize_label_schema(self.label_schema)
        self.delta_format = self.delta_format.strip()
        if not self.delta_format:
            raise ValueError("delta_format must not be empty.")
        if not self._has_update_material():
            raise ValueError(
                "PEFT-classifier update requires artifact refs or inline deltas."
            )
        if self.classifier_head_weight_deltas is not None:
            validate_label_vector_mapping(
                self.classifier_head_weight_deltas,
                labels=tuple(self.label_schema),
                field_name="classifier_head_weight_deltas",
            )
            self.classifier_head_bias_deltas = normalize_label_scalar_mapping(
                self.classifier_head_bias_deltas,
                labels=tuple(self.label_schema),
                field_name="classifier_head_bias_deltas",
            )
        if self.peft_parameter_deltas is not None:
            validate_non_empty_vector_mapping(
                self.peft_parameter_deltas,
                field_name="peft_parameter_deltas",
            )
        if self.partitioned_deltas is not None:
            self.partitioned_deltas = self._normalize_partitioned_deltas()
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label_schema)

    @property
    def peft_adapter_name(self) -> str:
        return self.peft_adapter_config.peft_adapter_name

    def _has_update_material(self) -> bool:
        return any(
            (
                self.peft_adapter_delta_artifact_ref,
                self.classifier_head_delta_artifact_ref,
                self.peft_parameter_deltas,
                self.classifier_head_weight_deltas,
                self.classifier_head_bias_deltas,
                self.partitioned_deltas,
                self.partitioned_deltas_artifact_ref,
            )
        )

    def _normalize_partitioned_deltas(
        self,
    ) -> dict[str, PeftClassifierPartitionDeltaPayload]:
        if self.partitioned_deltas is None:
            return {}
        labels = tuple(self.label_schema)
        normalized: dict[str, PeftClassifierPartitionDeltaPayload] = {}
        for raw_name, partition in self.partitioned_deltas.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError(
                    "partitioned_deltas partition names must not be empty."
                )
            if name in normalized:
                raise ValueError(f"Duplicate partitioned_deltas partition: {name}")
            if partition.classifier_head_weight_deltas:
                validate_label_vector_mapping(
                    partition.classifier_head_weight_deltas,
                    labels=labels,
                    field_name=f"partitioned_deltas.{name}.classifier_head_weight_deltas",
                )
            partition.classifier_head_bias_deltas = normalize_label_scalar_mapping(
                partition.classifier_head_bias_deltas,
                labels=labels,
                field_name=f"partitioned_deltas.{name}.classifier_head_bias_deltas",
            )
            normalized[name] = partition
        return normalized

    def l2_norm(self) -> float:
        if self.delta_l2_norm is not None:
            return self.delta_l2_norm
        squared_norm = 0.0
        has_primary_delta_material = False
        if self.peft_parameter_deltas is not None:
            has_primary_delta_material = True
            squared_norm += squared_vector_mapping_norm(self.peft_parameter_deltas)
        if self.classifier_head_weight_deltas is not None:
            has_primary_delta_material = True
            squared_norm += squared_vector_mapping_norm(
                self.classifier_head_weight_deltas
            )
        if self.classifier_head_bias_deltas:
            has_primary_delta_material = True
            squared_norm += sum(
                float(value) * float(value)
                for value in self.classifier_head_bias_deltas.values()
            )
        if not has_primary_delta_material and self.partitioned_deltas is not None:
            squared_norm += sum(
                partition.squared_l2_norm()
                for partition in self.partitioned_deltas.values()
            )
        if (
            not has_primary_delta_material
            and self.partitioned_deltas is None
            and self.partitioned_deltas_artifact_ref is not None
        ):
            raise ValueError(
                "partitioned_deltas_artifact_ref updates require delta_l2_norm for "
                "metadata-only norm calculation."
            )
        return math.sqrt(squared_norm)


PeftClassifierStatePayload = PeftClassifierAdapterStatePayload
PeftClassifierDeltaPayload = PeftClassifierAdapterUpdatePayload
PeftClassifierState = PeftClassifierStatePayload
PeftClassifierDelta = PeftClassifierDeltaPayload
