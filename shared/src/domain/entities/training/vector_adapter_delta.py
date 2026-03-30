"""로컬 학습이 만든 경량 adapter delta."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class VectorAdapterDelta:
    """현재 concrete 구현인 diagonal scale adapter update."""

    schema_version: str
    model_id: str
    base_model_revision: str
    training_scope: str
    dimension_deltas: list[float]
    example_count: int
    mean_confidence: float
    created_at: datetime | None = None
    mean_margin: float | None = None
    label_counts: dict[str, int] = field(default_factory=dict)
    adapter_kind: str = "diagonal_scale"

    @property
    def embedding_dim(self) -> int:
        """delta가 적용되는 임베딩 차원을 반환한다."""
        return len(self.dimension_deltas)

    def l2_norm(self) -> float:
        """delta 벡터의 L2 norm을 반환한다."""
        return math.sqrt(sum(value * value for value in self.dimension_deltas))
