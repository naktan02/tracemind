"""로컬 inference 이벤트 entity 모음."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class QueryEvent:
    """점수 계산 전 로컬에서 관측된 단일 query."""

    query_id: str
    text: str
    occurred_at: datetime
    locale: str
    source_type: str


@dataclass(slots=True)
class ScoredEvent:
    """번역, 임베딩, 카테고리 점수 계산을 거친 query event."""

    query_id: str
    occurred_at: datetime
    translated_text: str | None
    embedding_model_id: str
    translation_model_id: str | None
    category_scores: dict[str, float] = field(default_factory=dict)
