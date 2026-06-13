"""Captured text retention / capacity purge lifecycle service."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agent.src.infrastructure.repositories.captured_text.repository import (
    CapturedTextRepository,
)

CAPTURED_TEXT_RETENTION_DAYS_ENV = "TRACEMIND_CAPTURED_TEXT_RETENTION_DAYS"
CAPTURED_TEXT_MAX_RECORDS_ENV = "TRACEMIND_CAPTURED_TEXT_MAX_RECORDS"
DEFAULT_CAPTURED_TEXT_RETENTION_DAYS = 3
DEFAULT_CAPTURED_TEXT_MAX_RECORDS = 500


@dataclass(slots=True)
class CapturedTextLifecycleConfig:
    """agent 로컬 captured text retention / purge 정책."""

    retention_days: int = DEFAULT_CAPTURED_TEXT_RETENTION_DAYS
    max_records: int = DEFAULT_CAPTURED_TEXT_MAX_RECORDS

    def __post_init__(self) -> None:
        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive.")
        if self.max_records <= 0:
            raise ValueError("max_records must be positive.")


@dataclass(slots=True)
class CapturedTextPurgeResult:
    """retention / purge 실행 결과 요약."""

    deleted_by_retention: int = 0
    deleted_by_capacity: int = 0

    @property
    def deleted_total(self) -> int:
        return self.deleted_by_retention + self.deleted_by_capacity


@dataclass(slots=True)
class CapturedTextLifecycleService:
    """captured text raw event lifecycle을 config-driven으로 관리한다."""

    config: CapturedTextLifecycleConfig = field(
        default_factory=CapturedTextLifecycleConfig
    )

    def purge(
        self,
        *,
        repository: CapturedTextRepository,
        as_of: datetime | None = None,
    ) -> CapturedTextPurgeResult:
        effective_as_of = as_of or datetime.now(tz=timezone.utc)
        cutoff = effective_as_of - timedelta(days=self.config.retention_days)
        deleted_by_retention = repository.delete_older_than(cutoff=cutoff)
        deleted_by_capacity = repository.delete_oldest_excess(
            keep_latest=self.config.max_records
        )
        return CapturedTextPurgeResult(
            deleted_by_retention=deleted_by_retention,
            deleted_by_capacity=deleted_by_capacity,
        )


def build_captured_text_lifecycle_service_from_env(
    *,
    environ: Mapping[str, str] | None = None,
) -> CapturedTextLifecycleService:
    """환경변수에서 captured text 개발용 retention 정책을 조립한다."""

    effective_environ = os.environ if environ is None else environ
    return CapturedTextLifecycleService(
        config=CapturedTextLifecycleConfig(
            retention_days=_env_int(
                effective_environ,
                CAPTURED_TEXT_RETENTION_DAYS_ENV,
                DEFAULT_CAPTURED_TEXT_RETENTION_DAYS,
            ),
            max_records=_env_int(
                effective_environ,
                CAPTURED_TEXT_MAX_RECORDS_ENV,
                DEFAULT_CAPTURED_TEXT_MAX_RECORDS,
            ),
        )
    )


def _env_int(environ: Mapping[str, str], key: str, default: int) -> int:
    value = environ.get(key, "").strip()
    return int(value) if value else default
