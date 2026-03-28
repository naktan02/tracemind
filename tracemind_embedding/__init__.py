"""공통 임베딩 어댑터 패키지."""

from tracemind_embedding.base import EmbeddingAdapter, EmbeddingAdapterSpec
from tracemind_embedding.factory import EmbeddingAdapterFactory

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingAdapterFactory",
    "EmbeddingAdapterSpec",
]
