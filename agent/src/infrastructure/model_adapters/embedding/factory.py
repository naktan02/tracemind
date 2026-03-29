"""임베딩 어댑터 팩토리.

Hydra instantiate 외에도 문자열 기반 백엔드 선택이 필요한 스크립트용.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.src.infrastructure.model_adapters.embedding.base import EmbeddingAdapter


@dataclass(slots=True, frozen=True)
class EmbeddingAdapterSpec:
    """어댑터 생성에 필요한 설정을 묶는 값 객체."""

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


class EmbeddingAdapterFactory:
    """백엔드 문자열 기반 어댑터 팩토리."""

    _BACKENDS = ("hash_debug", "transformers_mxbai")

    @classmethod
    def supported_backends(cls) -> tuple[str, ...]:
        return cls._BACKENDS

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> Any:
        """spec으로부터 적절한 어댑터 인스턴스를 생성한다."""
        if spec.backend == "hash_debug":
            from agent.src.infrastructure.model_adapters.embedding.hash_debug import (
                HashDebugEmbeddingAdapter,
            )

            return HashDebugEmbeddingAdapter(dim=spec.hash_dim)

        if spec.backend == "transformers_mxbai":
            from agent.src.infrastructure.model_adapters.embedding.mxbai import (
                MxbaiEmbeddingAdapter,
            )

            return MxbaiEmbeddingAdapter(
                model_id=spec.model_id,
                revision=spec.revision,
                device=spec.device,
                normalize_embeddings=spec.normalize_embeddings,
                batch_size=spec.batch_size,
                cache_dir=spec.cache_dir,
                task_prefix=spec.task_prefix,
                local_files_only=spec.local_files_only,
            )

        raise ValueError(
            f"지원되지 않는 백엔드: '{spec.backend}'. "
            f"지원 목록: {cls._BACKENDS}"
        )
