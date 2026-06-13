"""텍스트 전처리 서비스."""

from dataclasses import dataclass


@dataclass(slots=True)
class PreprocessService:
    """번역 전에 결정적인 텍스트 정규화를 적용한다."""

    strip_whitespace: bool = True
    collapse_internal_spaces: bool = True

    def normalize(self, text: str) -> str:
        normalized = text.strip() if self.strip_whitespace else text
        if self.collapse_internal_spaces:
            normalized = " ".join(normalized.split())
        return normalized
