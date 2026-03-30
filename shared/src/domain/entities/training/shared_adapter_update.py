"""공통 shared adapter update 프로토콜."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class SharedAdapterUpdate(Protocol):
    """공통 shared adapter update가 제공해야 하는 최소 인터페이스."""

    schema_version: str
    adapter_kind: str
    model_id: str
    base_model_revision: str
    training_scope: str
    example_count: int
    mean_confidence: float
    mean_margin: float | None
    created_at: datetime | None

    @property
    def embedding_dim(self) -> int:
        """delta가 적용되는 임베딩 차원을 반환한다."""

    def l2_norm(self) -> float:
        """update 파라미터의 L2 norm을 반환한다."""
