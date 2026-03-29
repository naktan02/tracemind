"""관리자 라벨 query dataset entity 모음."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class LabeledQuery:
    """prototype 생성과 평가에 쓰는 수동 라벨 query 하나."""

    query_id: str
    text: str
    raw_label_scheme: str
    raw_label: str
    mapped_label_4: str
    locale: str
    annotation_source: str
    approved_by: str | None
    created_at: datetime


@dataclass(slots=True)
class LabeledQuerySet:
    """Git 추적 raw data 바깥에서 관리하는 버전형 labeled query 묶음."""

    dataset_id: str
    mapping_version: str
    queries: list[LabeledQuery] = field(default_factory=list)
