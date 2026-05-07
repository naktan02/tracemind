"""Classifier-head shared adapter contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field, model_validator

from shared.src.contracts.common_types import TrainingScope

from .base import (
    CLASSIFIER_HEAD_DELTA_V1,
    CLASSIFIER_HEAD_STATE_V1,
    AdapterKind,
    ClassifierHeadDeltaSchemaVersion,
    ClassifierHeadStateSchemaVersion,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)


class ClassifierHeadAdapterStatePayload(SharedAdapterStatePayload):
    """고정 임베딩 위 category별 선형 분류 head를 공유하는 family state."""

    schema_version: ClassifierHeadStateSchemaVersion = Field(
        default=CLASSIFIER_HEAD_STATE_V1,
        description="Classifier head state payload contract 버전.",
    )
    label_weights: dict[str, list[float]] = Field(
        description="카테고리별 선형 head weight 벡터."
    )
    label_biases: dict[str, float] = Field(
        default_factory=dict,
        description="카테고리별 bias 항.",
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
        updated_at,
    ) -> "ClassifierHeadAdapterStatePayload":
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


class ClassifierHeadAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """Classifier-head family update payload."""

    schema_version: ClassifierHeadDeltaSchemaVersion = Field(
        default=CLASSIFIER_HEAD_DELTA_V1,
        description="Classifier head update payload contract 버전.",
    )
    label_weight_deltas: dict[str, list[float]] = Field(
        description="카테고리별 weight delta 벡터."
    )
    label_bias_deltas: dict[str, float] = Field(default_factory=dict)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    mean_margin: float | None = None
    label_counts: dict[str, int] = Field(default_factory=dict)

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
        squared_weight_norm = sum(
            float(value) * float(value)
            for deltas in self.label_weight_deltas.values()
            for value in deltas
        )
        squared_bias_norm = sum(
            float(value) * float(value) for value in self.label_bias_deltas.values()
        )
        return math.sqrt(squared_weight_norm + squared_bias_norm)


ClassifierHeadStatePayload = ClassifierHeadAdapterStatePayload
ClassifierHeadDeltaPayload = ClassifierHeadAdapterUpdatePayload
ClassifierHeadState = ClassifierHeadStatePayload
ClassifierHeadDelta = ClassifierHeadDeltaPayload
