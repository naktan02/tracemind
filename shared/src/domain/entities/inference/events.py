"""로컬 inference 이벤트 entity 모음."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

JsonScalar = str | int | float | bool | None


@dataclass(slots=True)
class QueryEvent:
    """점수 계산 전 로컬에서 관측된 단일 query."""

    query_id: str
    text: str
    occurred_at: datetime
    locale: str
    source_type: str


@dataclass(slots=True)
class AnalysisEvent:
    """method-agnostic 로컬 분석 결과.

    classifier, prototype, hybrid scorer 모두 같은 analysis event로 저장한다.
    """

    analysis_id: str
    source_event_id: str
    occurred_at: datetime
    translated_text: str | None
    scorer_family: str
    scorer_name: str
    model_revision: str
    confidence_kind: str
    embedding_model_id: str | None
    translation_model_id: str | None
    category_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, JsonScalar] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredEvent:
    """기존 호출부 호환용 scored event projection."""

    query_id: str
    occurred_at: datetime
    translated_text: str | None
    embedding_model_id: str
    translation_model_id: str | None
    category_scores: dict[str, float] = field(default_factory=dict)
