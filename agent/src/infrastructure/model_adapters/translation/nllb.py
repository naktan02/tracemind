"""NLLB 번역 어댑터."""

from __future__ import annotations

from collections.abc import Sequence
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
    cache_dir: str | None = None
    local_files_only: bool = False
    _resolved_device: str = field(init=False, repr=False)

    _MODEL_CACHE: ClassVar[
        dict[tuple[str, str, str | None, bool, str], tuple[Any, Any]]
    ] = {}

    def __post_init__(self) -> None:
        self._resolved_device = resolve_runtime_device(self.device)

    def resolved_device(self) -> str:
        """실행 시 사용할 장치를 정규화한다."""
        return self._resolved_device

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
                str(text)
                for text in texts[start_index : start_index + self.batch_size]
            ]
            tokenizer.src_lang = self.source_lang
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self._resolved_device)
                for key, value in encoded.items()
            }
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=self.max_new_tokens,
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
        )
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        AutoModelForSeq2SeqLM, AutoTokenizer = _require_transformer_seq2seq_stack()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        ).to(self._resolved_device)
        model.eval()
        self._MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    def _resolve_target_bos_token_id(self, tokenizer: Any) -> int:
        lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and self.target_lang in lang_code_to_id:
            return int(lang_code_to_id[self.target_lang])
        return int(tokenizer.convert_tokens_to_ids(self.target_lang))


def _require_transformer_seq2seq_stack() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "NLLB translation requires transformers to be installed. "
            "Example: uv sync --extra experiments"
        ) from exc
    return AutoModelForSeq2SeqLM, AutoTokenizer
