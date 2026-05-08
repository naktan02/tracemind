"""LoRA-classifier shared adapter contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import (
    LORA_CLASSIFIER_DELTA_V1,
    LORA_CLASSIFIER_STATE_V1,
    AdapterKind,
    LoraClassifierDeltaSchemaVersion,
    LoraClassifierStateSchemaVersion,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
    normalize_label_scalar_mapping,
    normalize_label_schema,
    squared_vector_mapping_norm,
    validate_label_vector_mapping,
    validate_non_empty_vector_mapping,
)

LORA_CLASSIFIER_ADAPTER_KIND = AdapterKind.LORA_CLASSIFIER.value
LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT = "lora_classifier_update"
LORA_CLASSIFIER_ACCEPTED_UPDATE_PAYLOAD_FORMATS = (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
)


class LoraClassifierBackbonePayload(BaseModel):
    """LoRA-classifier가 결합되는 frozen text backbone/tokenizer 식별자."""

    model_config = ConfigDict(extra="forbid")

    backbone_model_id: str = Field(description="Frozen backbone model id.")
    backbone_revision: str = Field(description="Frozen backbone revision.")
    tokenizer_model_id: str = Field(description="Tokenizer model id.")
    tokenizer_revision: str = Field(description="Tokenizer revision.")
    pooling: str = Field(default="mean")
    max_length: int = Field(gt=0)
    task_prefix: str = Field(default="")


class LoraClassifierConfigPayload(BaseModel):
    """LoRA/RSLoRA adapter config snapshot."""

    model_config = ConfigDict(extra="forbid")

    peft_adapter_name: str = Field(default="lora")
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    bias: str = Field(default="none")
    target_modules: str | list[str]
    use_rslora: bool = Field(default=False)

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
    """LoRA adapter와 classifier head를 함께 배포하는 shared family state."""

    schema_version: LoraClassifierStateSchemaVersion = Field(
        default=LORA_CLASSIFIER_STATE_V1,
        description="LoRA-classifier state payload contract 버전.",
    )
    adapter_kind: str = Field(default=LORA_CLASSIFIER_ADAPTER_KIND)
    backbone: LoraClassifierBackbonePayload
    lora_config: LoraClassifierConfigPayload
    label_schema: list[str]
    lora_adapter_artifact_ref: str | None = None
    classifier_head_artifact_ref: str | None = None
    artifact_format: str = "artifact_ref"

    @model_validator(mode="after")
    def _validate_lora_classifier_state(
        self,
    ) -> "LoraClassifierAdapterStatePayload":
        self.label_schema = normalize_label_schema(self.label_schema)
        if not self.artifact_format.strip():
            raise ValueError("artifact_format must not be empty.")
        self.artifact_format = self.artifact_format.strip()
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label_schema)

    def apply(self, embedding: Sequence[float]) -> list[float]:
        vector = [float(value) for value in embedding]
        if not vector:
            raise ValueError("LoRA-classifier input embedding must not be empty.")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("LoRA-classifier input embedding norm must be non-zero.")
        return [value / norm for value in vector]


class LoraClassifierAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """LoRA-classifier family update payload."""

    schema_version: LoraClassifierDeltaSchemaVersion = Field(
        default=LORA_CLASSIFIER_DELTA_V1,
        description="LoRA-classifier update payload contract 버전.",
    )
    adapter_kind: str = Field(default=LORA_CLASSIFIER_ADAPTER_KIND)
    canonical_update_payload_format: ClassVar[str] = (
        LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
    )
    accepted_update_payload_formats: ClassVar[tuple[str, ...]] = (
        LORA_CLASSIFIER_ACCEPTED_UPDATE_PAYLOAD_FORMATS
    )
    backbone: LoraClassifierBackbonePayload
    lora_config: LoraClassifierConfigPayload
    label_schema: list[str]
    lora_delta_artifact_ref: str | None = None
    classifier_head_delta_artifact_ref: str | None = None
    lora_parameter_deltas: dict[str, list[float]] | None = None
    classifier_head_weight_deltas: dict[str, list[float]] | None = None
    classifier_head_bias_deltas: dict[str, float] = Field(default_factory=dict)
    delta_format: str = "artifact_ref"
    mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_margin: float | None = None
    label_counts: dict[str, int] = Field(default_factory=dict)
    delta_l2_norm: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_lora_classifier_delta(
        self,
    ) -> "LoraClassifierAdapterUpdatePayload":
        self.label_schema = normalize_label_schema(self.label_schema)
        self.delta_format = self.delta_format.strip()
        if not self.delta_format:
            raise ValueError("delta_format must not be empty.")
        if not self._has_update_material():
            raise ValueError(
                "LoRA-classifier update requires artifact refs or inline deltas."
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
        if self.lora_parameter_deltas is not None:
            validate_non_empty_vector_mapping(
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
        if self.delta_l2_norm is not None:
            return self.delta_l2_norm
        squared_norm = 0.0
        if self.lora_parameter_deltas is not None:
            squared_norm += squared_vector_mapping_norm(self.lora_parameter_deltas)
        if self.classifier_head_weight_deltas is not None:
            squared_norm += squared_vector_mapping_norm(
                self.classifier_head_weight_deltas
            )
        squared_norm += sum(
            float(value) * float(value)
            for value in self.classifier_head_bias_deltas.values()
        )
        return math.sqrt(squared_norm)


LoraClassifierStatePayload = LoraClassifierAdapterStatePayload
LoraClassifierDeltaPayload = LoraClassifierAdapterUpdatePayload
LoraClassifierState = LoraClassifierStatePayload
LoraClassifierDelta = LoraClassifierDeltaPayload
