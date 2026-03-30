"""공통 shared adapter 상태 프로토콜."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol


class SharedAdapterState(Protocol):
    """공통 shared adapter 상태가 제공해야 하는 최소 인터페이스."""

    schema_version: str
    adapter_kind: str
    model_id: str
    model_revision: str
    training_scope: str
    updated_at: datetime

    @property
    def embedding_dim(self) -> int:
        """현재 adapter가 기대하는 임베딩 차원을 반환한다."""

    def apply(self, embedding: Sequence[float]) -> list[float]:
        """임베딩에 adapter 변환을 적용한다."""
