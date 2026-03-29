"""전역 공통 임베딩 공간 위에 얹히는 경량 adapter 상태."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class VectorAdapterState:
    """차원별 scale로 임베딩 공간을 미세 조정하는 경량 상태."""

    schema_version: str
    model_id: str
    model_revision: str
    training_scope: str
    dimension_scales: list[float]
    updated_at: datetime

    @classmethod
    def identity(
        cls,
        *,
        model_id: str,
        model_revision: str,
        training_scope: str,
        embedding_dim: int,
        updated_at: datetime,
        schema_version: str = "vector_adapter_state.v1",
    ) -> "VectorAdapterState":
        """아무 보정도 하지 않는 초기 adapter 상태를 만든다."""
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        return cls(
            schema_version=schema_version,
            model_id=model_id,
            model_revision=model_revision,
            training_scope=training_scope,
            dimension_scales=[1.0] * embedding_dim,
            updated_at=updated_at,
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
