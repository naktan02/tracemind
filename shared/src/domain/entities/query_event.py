"""원본 로컬 query event."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class QueryEvent:
    """점수 계산 전 로컬에서 관측된 단일 query."""

    query_id: str
    text: str
    occurred_at: datetime
    locale: str
    source_type: str
