"""Query buffer retention / purge lifecycle service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRepository,
)


@dataclass(slots=True)
class QueryBufferLifecycleConfig:
    """agent 로컬 query buffer retention / purge 정책."""

    retention_days: int = 30
    max_records: int = 5000

    def __post_init__(self) -> None:
        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive.")
        if self.max_records <= 0:
            raise ValueError("max_records must be positive.")


@dataclass(slots=True)
class QueryBufferPurgeResult:
    """retention / purge 실행 결과 요약."""

    deleted_by_retention: int = 0
    deleted_by_capacity: int = 0

    @property
    def deleted_total(self) -> int:
        return self.deleted_by_retention + self.deleted_by_capacity


@dataclass(slots=True)
class QueryBufferLifecycleService:
    """query buffer raw text lifecycle을 config-driven으로 관리한다."""

    config: QueryBufferLifecycleConfig = field(
        default_factory=QueryBufferLifecycleConfig
    )

    def purge(
        self,
        *,
        repository: QueryBufferRepository,
        as_of: datetime | None = None,
    ) -> QueryBufferPurgeResult:
        effective_as_of = as_of or datetime.now(tz=timezone.utc)
        cutoff = effective_as_of - timedelta(days=self.config.retention_days)
        deleted_by_retention = repository.delete_older_than(cutoff=cutoff)
        deleted_by_capacity = repository.delete_oldest_excess(
            keep_latest=self.config.max_records
        )
        return QueryBufferPurgeResult(
            deleted_by_retention=deleted_by_retention,
            deleted_by_capacity=deleted_by_capacity,
        )
