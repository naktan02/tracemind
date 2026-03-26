"""MXBAI 임베딩 어댑터 자리표시자."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class MxbaiEmbeddingAdapter:
    """`mixedbread-ai/mxbai-embed-large-v1`용 어댑터 설정."""

    model_id: str
    revision: str = "main"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 16
    cache_dir: str | None = None
    task_prefix: str = ""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError("MXBAI inference wiring is not implemented yet.")
