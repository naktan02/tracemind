"""공용 임베딩 어댑터 포트."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """텍스트 배치를 임베딩 벡터 리스트로 변환하는 최소 포트."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """텍스트 목록을 임베딩한다."""
        ...
