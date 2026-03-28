"""임베딩 어댑터 팩토리."""

from __future__ import annotations

from tracemind_embedding.adapters.hash_debug import HashDebugEmbeddingAdapter
from tracemind_embedding.adapters.mxbai import MxbaiEmbeddingAdapter
from tracemind_embedding.base import EmbeddingAdapter, EmbeddingAdapterSpec


class EmbeddingAdapterFactory:
    """백엔드 이름으로 임베딩 어댑터를 생성한다."""

    _BACKEND_ALIASES = {
        "transformers_mxbai": "mxbai_large",
    }

    @classmethod
    def supported_backends(cls) -> tuple[str, ...]:
        """CLI나 설정 파일에서 허용할 백엔드 이름 목록."""
        return ("hash_debug", "mxbai_large", *cls._BACKEND_ALIASES)

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> EmbeddingAdapter:
        """설정 스펙으로부터 임베딩 어댑터를 만든다."""
        backend = cls._BACKEND_ALIASES.get(spec.backend, spec.backend)
        if backend == "hash_debug":
            return HashDebugEmbeddingAdapter(
                vector_dim=spec.hash_dim,
                normalize_embeddings=spec.normalize_embeddings,
            )
        if backend == "mxbai_large":
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

        raise ValueError(f"Unsupported embedding backend: {spec.backend}")
