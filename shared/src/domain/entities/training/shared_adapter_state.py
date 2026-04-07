"""공통 shared adapter 상태 프로토콜."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(slots=True)
class IdentitySharedAdapterState:
    """추가 보정 없이 임베딩을 그대로 정규화해 통과시키는 공용 identity state."""

    model_id: str
    model_revision: str
    training_scope: str
    embedding_dim: int
    updated_at: datetime
    schema_version: str = "shared_adapter_state.identity.v1"
    adapter_kind: str = "identity"

    def apply(self, embedding: Sequence[float]) -> list[float]:
        if len(embedding) != self.embedding_dim:
            raise ValueError("Embedding dimension does not match adapter state.")
        norm = math.sqrt(sum(float(value) * float(value) for value in embedding))
        if norm == 0.0:
            raise ValueError("Adapter-transformed embedding norm must be non-zero.")
        return [float(value) / norm for value in embedding]
