"""Cohort 집계 서비스 자리표시자."""

from dataclasses import dataclass


@dataclass(slots=True)
class AggregationService:
    """window summary를 cohort 단위 통계로 집계한다."""

    min_support: int = 30
