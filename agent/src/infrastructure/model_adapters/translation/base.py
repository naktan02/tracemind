"""번역 어댑터 프로토콜."""

from collections.abc import Sequence
from typing import Protocol


class TranslationAdapter(Protocol):
    """교체 가능한 번역 모델 어댑터 인터페이스."""

    def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """텍스트 배치를 목표 언어로 번역한다."""
