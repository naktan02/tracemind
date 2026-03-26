"""Cohort 규범 통계."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class CategoryNorm:
    """카테고리 하나에 대한 cohort 단위 통계."""

    mean: float | None = None
    sigma: float | None = None
    median: float | None = None
    mad: float | None = None
    prevalence: float | None = None


@dataclass(slots=True)
class NormPack:
    """로컬 agent가 내려받는 또래 기준 통계."""

    schema_version: str
    cohort_key: str
    pack_version: str
    generated_at: datetime
    min_support: int
    categories: dict[str, CategoryNorm] = field(default_factory=dict)
