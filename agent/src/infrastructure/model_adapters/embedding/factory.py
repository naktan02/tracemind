"""임베딩 어댑터 팩토리.

Hydra instantiate 외에도 문자열 기반 백엔드 선택이 필요한 스크립트용.
"""

from __future__ import annotations

from shared.src.domain.services.embedding_adapter import EmbeddingAdapter
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


class EmbeddingAdapterFactory:
    """백엔드 문자열 기반 어댑터 팩토리."""

    _BACKENDS = ("hash_debug", "transformers_mxbai")

    @classmethod
    def supported_backends(cls) -> tuple[str, ...]:
        return cls._BACKENDS

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> EmbeddingAdapter:
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
            f"지원되지 않는 백엔드: '{spec.backend}'. 지원 목록: {cls._BACKENDS}"
        )
