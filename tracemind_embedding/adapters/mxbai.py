"""MXBAI 임베딩 어댑터."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tracemind_runtime import resolve_runtime_device


@dataclass(slots=True)
class _TransformersRuntime:
    """Transformers 추론에 필요한 객체 묶음."""

    torch: Any
    tokenizer: Any
    model: Any


@dataclass(slots=True)
class MxbaiEmbeddingAdapter:
    """`mixedbread-ai/mxbai-embed-large-v1` 계열 임베딩 어댑터."""

    model_id: str
    revision: str = "main"
    device: str = "auto"
    normalize_embeddings: bool = True
    batch_size: int = 16
    cache_dir: str | None = None
    task_prefix: str = ""
    local_files_only: bool = False
    _sentence_transformer_model: Any | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _transformers_runtime: _TransformersRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _runtime_device: str | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        prefixed_texts = [f"{self.task_prefix}{text}" for text in texts]
        sentence_transformer_class = self._load_sentence_transformer_class()
        if sentence_transformer_class is not None:
            model = self._get_sentence_transformer_model(sentence_transformer_class)
            encoded = model.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=False,
                show_progress_bar=False,
            )
            return self._coerce_rows(encoded)

        runtime = self._get_transformers_runtime()
        return self._encode_with_transformers(prefixed_texts, runtime)

    def _get_runtime_device(self) -> str:
        if self._runtime_device is None:
            self._runtime_device = resolve_runtime_device(self.device)
        return self._runtime_device

    def _load_sentence_transformer_class(self) -> type[Any] | None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError:
            return None
        return getattr(module, "SentenceTransformer")

    def _get_sentence_transformer_model(
        self,
        sentence_transformer_class: type[Any],
    ) -> Any:
        if self._sentence_transformer_model is None:
            kwargs: dict[str, Any] = {"revision": self.revision}
            if self.cache_dir is not None:
                kwargs["cache_folder"] = self.cache_dir
            if self.local_files_only:
                kwargs["local_files_only"] = True
            model = sentence_transformer_class(self.model_id, **kwargs)
            if hasattr(model, "to"):
                model.to(self._get_runtime_device())
            self._sentence_transformer_model = model
        return self._sentence_transformer_model

    def _get_transformers_runtime(self) -> _TransformersRuntime:
        if self._transformers_runtime is None:
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
            except ImportError as exc:
                raise RuntimeError(
                    "mxbai_large backend requires sentence_transformers "
                    "or torch+transformers."
                ) from exc

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            model = transformers.AutoModel.from_pretrained(
                self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            model.to(self._get_runtime_device())
            model.eval()
            self._transformers_runtime = _TransformersRuntime(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
            )

        return self._transformers_runtime

    def _encode_with_transformers(
        self,
        texts: list[str],
        runtime: _TransformersRuntime,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self._batched(texts):
            encoded = runtime.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = self._move_tensors_to_device(encoded)
            with runtime.torch.no_grad():
                outputs = runtime.model(**encoded)

            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = outputs.last_hidden_state * attention_mask
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            if self.normalize_embeddings:
                pooled = runtime.torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(self._coerce_rows(pooled.cpu()))

        return embeddings

    def _batched(self, texts: list[str]) -> list[list[str]]:
        return [
            texts[index : index + self.batch_size]
            for index in range(0, len(texts), self.batch_size)
        ]

    def _move_tensors_to_device(
        self,
        encoded: Mapping[str, Any],
    ) -> dict[str, Any]:
        runtime_device = self._get_runtime_device()
        return {
            name: value.to(runtime_device) if hasattr(value, "to") else value
            for name, value in encoded.items()
        }

    @staticmethod
    def _coerce_rows(values: Any) -> list[list[float]]:
        rows = values.tolist() if hasattr(values, "tolist") else values
        return [
            [float(component) for component in row]
            for row in rows
        ]
