"""MXBAI sentence-transformers 기반 임베딩 어댑터."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.src.infrastructure.runtime import resolve_runtime_device


@dataclass
class MxbaiEmbeddingAdapter:
    """mixedbread-ai/mxbai-embed-large-v1 어댑터.

    Hydra instantiate로 생성할 수 있도록 dataclass 생성자 시그니처를 유지한다.
    모델 로딩은 첫 embed_texts() 호출 시 lazy하게 수행한다.
    """

    model_id: str = "mixedbread-ai/mxbai-embed-large-v1"
    revision: str = "main"
    device: str = "auto"
    normalize_embeddings: bool = True
    batch_size: int = 16
    cache_dir: str | None = None
    task_prefix: str = ""
    local_files_only: bool = False

    _model: Any = field(init=False, default=None, repr=False)
    _resolved_device: str = field(init=False, default="", repr=False)

    def _ensure_model(self) -> None:
        """모델이 아직 로드되지 않았으면 lazy하게 로드한다."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._resolved_device = resolve_runtime_device(self.device)
        self._model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device=self._resolved_device,
            cache_folder=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        print(
            f"[MxbaiEmbeddingAdapter] 모델 로드 완료: "
            f"model={self.model_id} device={self._resolved_device}",
            flush=True,
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """텍스트 배치를 임베딩한다.

        task_prefix가 설정돼 있으면 각 텍스트 앞에 붙인다.
        """
        self._ensure_model()

        input_texts = list(texts)
        if self.task_prefix:
            input_texts = [f"{self.task_prefix}{t}" for t in input_texts]

        embeddings = self._model.encode(
            input_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return [row.tolist() for row in embeddings]
