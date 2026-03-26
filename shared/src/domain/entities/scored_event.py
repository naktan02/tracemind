"""로컬 점수 계산 결과 event 표현."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ScoredEvent:
    """번역, 임베딩, 카테고리 점수 계산을 거친 query event."""

    query_id: str
    translated_text: str | None
    embedding_model_id: str
    translation_model_id: str | None
    category_scores: dict[str, float] = field(default_factory=dict)
