"""임베딩 어댑터 프로토콜."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """교체 가능한 임베딩 모델 어댑터 인터페이스."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """텍스트 배치를 공통 벡터 공간으로 임베딩한다."""
