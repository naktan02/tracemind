"""교체 가능한 임베딩 어댑터 프로토콜."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """임베딩 모델 어댑터 인터페이스.

    모든 임베딩 백엔드(mxbai, hash_debug 등)는 이 프로토콜을 구현한다.
    """

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """텍스트 배치를 임베딩 벡터 리스트로 변환한다."""
        ...
