"""재사용 가능한 backtranslation 서비스."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent.src.infrastructure.model_adapters.translation.nllb import (
    NllbTranslationAdapter,
)
from agent.src.services.translation_service import TranslationService


@dataclass(frozen=True, slots=True)
class BacktranslationCandidatePair:
    """원문 하나에 대한 strong candidate 두 개."""

    aug_0: str
    aug_1: str
    aug_0_pivot_lang: str
    aug_1_pivot_lang: str


@dataclass(frozen=True, slots=True)
class NllbBacktranslationService:
    """NLLB를 조합해 backtranslation candidate를 생성한다."""

    source_lang: str
    pivot_languages: tuple[str, str]
    model_id: str = "facebook/nllb-200-distilled-600M"
    revision: str = "main"
    device: str = "auto"
    batch_size: int = 8
    max_new_tokens: int = 256
    torch_dtype: str = "auto"
    cache_dir: str | None = None
    local_files_only: bool = False

    def build_candidate_pairs(
        self,
        *,
        texts: Sequence[str],
    ) -> list[BacktranslationCandidatePair]:
        if len(self.pivot_languages) != 2:
            raise ValueError(
                "Strict USB-style FixMatch augmentation requires exactly two "
                "pivot languages."
            )
        if not texts:
            return []

        pivot_outputs: dict[str, list[str]] = {}
        for pivot_lang in self.pivot_languages:
            pivot_outputs[pivot_lang] = self._backtranslate(
                texts=texts,
                pivot_lang=pivot_lang,
            )

        first_pivot, second_pivot = self.pivot_languages
        return [
            BacktranslationCandidatePair(
                aug_0=pivot_outputs[first_pivot][index],
                aug_1=pivot_outputs[second_pivot][index],
                aug_0_pivot_lang=first_pivot,
                aug_1_pivot_lang=second_pivot,
            )
            for index in range(len(texts))
        ]

    def _backtranslate(
        self,
        *,
        texts: Sequence[str],
        pivot_lang: str,
    ) -> list[str]:
        to_pivot = TranslationService(
            adapter=NllbTranslationAdapter(
                model_id=self.model_id,
                revision=self.revision,
                source_lang=self.source_lang,
                target_lang=pivot_lang,
                device=self.device,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        )
        to_source = TranslationService(
            adapter=NllbTranslationAdapter(
                model_id=self.model_id,
                revision=self.revision,
                source_lang=pivot_lang,
                target_lang=self.source_lang,
                device=self.device,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        )
        pivot_texts = to_pivot.translate_batch(list(texts))
        return to_source.translate_batch(pivot_texts)
