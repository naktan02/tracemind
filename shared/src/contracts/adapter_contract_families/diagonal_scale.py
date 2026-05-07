"""Diagonal-scale shared adapter contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from pydantic import Field

from shared.src.contracts.common_types import TrainingScope

from .base import (
    VECTOR_ADAPTER_STATE_V1,
    AdapterKind,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)


class DiagonalScaleAdapterStatePayload(SharedAdapterStatePayload):
    """임베딩 각 차원에 곱하는 scale 벡터를 공유하는 adapter state."""

    dimension_scales: list[float] = Field(description="임베딩 차원별 전역 scale 벡터.")

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
        return len(self.dimension_scales)

    def apply(self, embedding: Sequence[float]) -> list[float]:
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


class DiagonalScaleAdapterUpdatePayload(SharedAdapterUpdatePayload):
    """Diagonal-scale adapter update payload."""

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
        description="Accepted example의 pseudo-label 분포.",
    )

    @property
    def embedding_dim(self) -> int:
        return len(self.dimension_deltas)

    def l2_norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.dimension_deltas))


VectorAdapterStatePayload = DiagonalScaleAdapterStatePayload
VectorAdapterDeltaPayload = DiagonalScaleAdapterUpdatePayload
VectorAdapterState = VectorAdapterStatePayload
VectorAdapterDelta = VectorAdapterDeltaPayload
