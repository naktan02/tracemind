"""Fixed-feature 분류용 feature space 생성."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


def build_feature_space(config: Mapping[str, Any]) -> Any:
    """Hydra leaf mapping에서 feature transformer를 생성한다."""

    name = str(config.get("name", ""))
    if name == "tfidf_word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(
                int(config.get("ngram_min", 1)),
                int(config.get("ngram_max", 2)),
            ),
            min_df=int(config.get("min_df", 2)),
            max_df=float(config.get("max_df", 0.95)),
            sublinear_tf=bool(config.get("sublinear_tf", True)),
            lowercase=bool(config.get("lowercase", True)),
            strip_accents=str(config.get("strip_accents", "unicode")),
        )
    if name == "frozen_embedding_mxbai":
        return FrozenSentenceEmbeddingFeatureSpace(
            model_id=str(config.get("model_id", "mixedbread-ai/mxbai-embed-large-v1")),
            revision=str(config.get("revision", "main")),
            device=str(config.get("device", "auto")),
            batch_size=int(config.get("batch_size", 16)),
            cache_dir=_optional_str(config.get("cache_dir")),
            task_prefix=str(config.get("task_prefix", "")),
            normalize_embeddings=bool(config.get("normalize_embeddings", True)),
            local_files_only=bool(config.get("local_files_only", False)),
            show_progress_bar=bool(config.get("show_progress_bar", True)),
        )
    raise ValueError(f"Unsupported fixed feature space: {name}")


class FrozenSentenceEmbeddingFeatureSpace(BaseEstimator, TransformerMixin):
    """sentence-transformers encoder를 고정 feature extractor로 쓰는 transformer."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str = "main",
        device: str = "auto",
        batch_size: int = 16,
        cache_dir: str | None = None,
        task_prefix: str = "",
        normalize_embeddings: bool = True,
        local_files_only: bool = False,
        show_progress_bar: bool = True,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.task_prefix = task_prefix
        self.normalize_embeddings = normalize_embeddings
        self.local_files_only = local_files_only
        self.show_progress_bar = show_progress_bar
        self._model: Any = None
        self._resolved_device = ""

    def fit(self, texts: list[str], labels: list[str] | None = None) -> Any:
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        input_texts = [str(text) for text in texts]
        if self.task_prefix:
            input_texts = [f"{self.task_prefix}{text}" for text in input_texts]
        embeddings = model.encode(
            input_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                "Frozen embedding feature space expected a 2D embedding matrix."
            )
        return matrix

    def fit_transform(
        self,
        texts: list[str],
        labels: list[str] | None = None,
        **fit_params: Any,
    ) -> np.ndarray:
        del labels, fit_params
        return self.transform(texts)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_model"] = None
        return state

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        sentence_transformer_cls = _load_sentence_transformer_class()
        self._resolved_device = _resolve_transformer_device(self.device)
        self._model = sentence_transformer_cls(
            self.model_id,
            revision=self.revision,
            device=self._resolved_device,
            cache_folder=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        return self._model


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_transformer_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "frozen_embedding_mxbai feature space requires sentence-transformers. "
            "Run `uv sync --extra dev --extra experiments` first."
        ) from error
    return SentenceTransformer
