"""번역 서비스."""

from dataclasses import dataclass

from src.infrastructure.model_adapters.translation.base import TranslationAdapter


@dataclass(slots=True)
class TranslationService:
    """모델 교체가 비즈니스 로직 바깥에 머물도록 번역 어댑터를 감싼다."""

    adapter: TranslationAdapter

    def translate_batch(self, texts: list[str]) -> list[str]:
        return self.adapter.translate_texts(texts)
