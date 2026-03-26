"""OPUS-MT 번역 어댑터 자리표시자."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class OpusTranslationAdapter:
    """가벼운 KO->EN 기준 모델용 어댑터 설정."""

    model_id: str
    source_lang: str
    target_lang: str
    revision: str = "main"
    device: str = "cpu"
    batch_size: int = 8
    max_new_tokens: int = 256
    cache_dir: str | None = None

    def translate_texts(self, texts: Sequence[str]) -> list[str]:
        raise NotImplementedError("OPUS inference wiring is not implemented yet.")
