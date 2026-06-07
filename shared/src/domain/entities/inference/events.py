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

    scorer backend 결과는 같은 analysis event로 저장한다.
    """

    query_id: str
    occurred_at: datetime
    translated_text: str | None
    embedding_model_id: str
    translation_model_id: str | None
    category_scores: dict[str, float] = field(default_factory=dict)
    analysis_id: str | None = None
    source_event_id: str | None = None
    scorer_family: str = "unknown"
    scorer_name: str = "unknown"
    model_revision: str = "unknown"
    confidence_kind: str = "unknown"
    metadata: dict[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.analysis_id is None:
            self.analysis_id = self.query_id
        if self.source_event_id is None:
            self.source_event_id = self.query_id
