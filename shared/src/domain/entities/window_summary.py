"""프라이버시 안전 요약 entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class CategoryStats:
    """하나의 summary window에 대한 카테고리별 집계 통계."""

    mean: float
    max: float
    count: int


@dataclass(slots=True)
class WindowSummary:
    """중앙 서버로 보내는 로컬 micro-batch 또는 rolling summary."""

    schema_version: str
    summary_id: str
    age_band: str
    batch_started_at: datetime
    batch_ended_at: datetime
    event_count: int
    category_stats: dict[str, CategoryStats] = field(default_factory=dict)
