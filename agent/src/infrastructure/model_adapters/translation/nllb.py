"""NLLB 번역 어댑터."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from agent.src.infrastructure.runtime import resolve_runtime_device


@dataclass(slots=True)
class NllbTranslationAdapter:
    """`facebook/nllb-200-distilled-600M`용 어댑터 설정."""

    model_id: str
    source_lang: str
    target_lang: str
    revision: str = "main"
    device: str = "auto"
    batch_size: int = 8
    max_new_tokens: int = 256
    torch_dtype: str = "auto"
    cache_dir: str | None = None
    local_files_only: bool = False
    _resolved_device: str = field(init=False, repr=False)
    _resolved_torch_dtype_name: str = field(init=False, repr=False)

    _MODEL_CACHE: ClassVar[
        dict[tuple[str, str, str | None, bool, str, str], tuple[Any, Any]]
    ] = {}

    def __post_init__(self) -> None:
        self._resolved_device = resolve_runtime_device(self.device)
        self._resolved_torch_dtype_name = self._resolve_torch_dtype_name()

    def resolved_device(self) -> str:
        """실행 시 사용할 장치를 정규화한다."""
        return self._resolved_device

    def resolved_torch_dtype(self) -> str:
        """실행 시 사용할 torch dtype 이름을 정규화한다."""
        return self._resolved_torch_dtype_name

    def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """텍스트 배치를 NLLB로 번역한다."""

        if not texts:
            return []

        import torch

        tokenizer, model = self._load_bundle()
        forced_bos_token_id = self._resolve_target_bos_token_id(tokenizer)

        translations: list[str] = []
        for start_index in range(0, len(texts), self.batch_size):
            batch_texts = [
                str(text) for text in texts[start_index : start_index + self.batch_size]
            ]
            tokenizer.src_lang = self.source_lang
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self._resolved_device) for key, value in encoded.items()
            }
            generation_config = None
            if getattr(model, "generation_config", None) is not None:
                generation_config = deepcopy(model.generation_config)
                generation_config.max_length = None
                generation_config.max_new_tokens = self.max_new_tokens
                generation_config.forced_bos_token_id = forced_bos_token_id
            with torch.no_grad():
                generate_kwargs = {**encoded}
                if generation_config is not None:
                    generate_kwargs["generation_config"] = generation_config
                else:
                    generate_kwargs["max_new_tokens"] = self.max_new_tokens
                    generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
                generated = model.generate(
                    **generate_kwargs,
                )
            translations.extend(
                tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            )
        return translations

    def _load_bundle(self) -> tuple[Any, Any]:
        cache_key = (
            self.model_id,
            self.revision,
            self.cache_dir,
            self.local_files_only,
            self._resolved_device,
            self._resolved_torch_dtype_name,
        )
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        torch_dtype = self._resolve_torch_dtype()
        AutoModelForSeq2SeqLM, AutoTokenizer = _require_transformer_seq2seq_stack()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        model_load_kwargs = {
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if torch_dtype is not None:
            model_load_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            **model_load_kwargs,
        ).to(self._resolved_device)
        model.eval()
        self._MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    def _resolve_target_bos_token_id(self, tokenizer: Any) -> int:
        lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and self.target_lang in lang_code_to_id:
            return int(lang_code_to_id[self.target_lang])
        return int(tokenizer.convert_tokens_to_ids(self.target_lang))

    def _resolve_torch_dtype_name(self) -> str:
        normalized = str(self.torch_dtype or "auto").strip().lower()
        if normalized in {"", "auto"}:
            if self._resolved_device.startswith("cuda"):
                return "float16"
            return "float32"
        if normalized in {"float16", "fp16", "half"}:
            return "float16"
        if normalized in {"bfloat16", "bf16"}:
            return "bfloat16"
        if normalized in {"float32", "fp32"}:
            return "float32"
        raise ValueError(
            "Unsupported NLLB torch_dtype. Use one of: auto, float16, "
            "bfloat16, float32."
        )

    def _resolve_torch_dtype(self) -> Any | None:
        import torch

        dtype_name = self._resolved_torch_dtype_name
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        return None


def _require_transformer_seq2seq_stack() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "NLLB translation requires transformers to be installed. "
            "Example: uv sync --extra experiments"
        ) from exc
    return AutoModelForSeq2SeqLM, AutoTokenizer
