"""임베딩 서비스."""

from dataclasses import dataclass

from shared.src.domain.services.embedding_adapter import EmbeddingAdapter


@dataclass(slots=True)
class EmbeddingService:
    """쿼리 임베딩 생성을 위해 임베딩 어댑터를 감싼다."""

    adapter: EmbeddingAdapter

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.adapter.embed_texts(texts)
