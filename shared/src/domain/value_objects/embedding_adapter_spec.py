"""임베딩 어댑터 생성 설정 값 객체."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EmbeddingAdapterSpec:
    """임베딩 backend 생성에 필요한 canonical 설정."""

    backend: str
    model_id: str = "mixedbread-ai/mxbai-embed-large-v1"
    revision: str = "main"
    device: str = "auto"
    batch_size: int = 16
    cache_dir: str | None = None
    task_prefix: str = ""
    normalize_embeddings: bool = True
    hash_dim: int = 256
    local_files_only: bool = False
