"""임베딩 어댑터 공통 타입."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """교체 가능한 임베딩 모델 어댑터 인터페이스."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """텍스트 배치를 공통 벡터 공간으로 임베딩한다."""


@dataclass(slots=True, frozen=True)
class EmbeddingAdapterSpec:
    """임베딩 어댑터 생성을 위한 공통 설정."""

    backend: str
    model_id: str
    revision: str = "main"
    device: str = "auto"
    normalize_embeddings: bool = True
    batch_size: int = 16
    cache_dir: str | None = None
    task_prefix: str = ""
    hash_dim: int = 256
    local_files_only: bool = False
