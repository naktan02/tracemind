"""CapturedTextEvent agent-local SQLite repository entrypoint."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.src.features.captured_text.storage import (
    analysis_job_store,
    event_store,
    generated_view_store,
    retention,
    view_job_store,
)
from agent.src.features.captured_text.storage.records import (
    CAPTURED_TEXT_ANALYSIS_STATUS_PENDING,
    CAPTURED_TEXT_VIEW_STATUS_DUPLICATE,
    CAPTURED_TEXT_VIEW_STATUS_PENDING,
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextAnalysisSourceRecord,
    CapturedTextGeneratedTrainingSourceRecord,
    CapturedTextGeneratedViewRecord,
    CapturedTextRecord,
)
from agent.src.features.captured_text.storage.schema import (
    ensure_captured_text_schema,
)
from agent.src.infrastructure.repositories.local_agent_database import (
    DEFAULT_AGENT_LOCAL_DB_PATH,
    connect_agent_local_db,
)

_DEFAULT_DB_PATH = DEFAULT_AGENT_LOCAL_DB_PATH


@dataclass(slots=True)
class CapturedTextRepository:
    """Captured text public repository API.

    이 class는 기존 caller가 쓰는 API를 유지하고, table별 SQL은 같은 package의
    store module이 소유한다.
    """

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            ensure_captured_text_schema(conn)

    def save(self, record: CapturedTextRecord) -> None:
        """event_id 기준으로 raw captured text event를 저장한다."""

        with self._connect() as conn:
            normalized = event_store.save_event(conn, record)
            view_job_store.upsert_job(
                conn,
                event_id=normalized.event_id,
                status=(
                    CAPTURED_TEXT_VIEW_STATUS_DUPLICATE
                    if normalized.duplicate_of_event_id is not None
                    else CAPTURED_TEXT_VIEW_STATUS_PENDING
                ),
                metadata={
                    "duplicate_of_event_id": normalized.duplicate_of_event_id,
                }
                if normalized.duplicate_of_event_id is not None
                else {},
            )

    def get(self, event_id: str) -> CapturedTextRecord | None:
        """단일 event_id의 저장 레코드를 반환한다."""

        with self._connect() as conn:
            return event_store.get_event(conn, event_id)

    def get_recent(self, *, limit: int = 50) -> list[CapturedTextRecord]:
        """최근 captured text event를 최신순으로 반환한다."""

        _require_positive_limit(limit)
        with self._connect() as conn:
            return event_store.get_recent_events(conn, limit=limit)

    def get_pending_view_generation(
        self,
        *,
        limit: int = 100,
    ) -> list[CapturedTextRecord]:
        """view generation을 기다리는 raw event를 오래된 순서로 반환한다."""

        _require_positive_limit(limit)
        with self._connect() as conn:
            return view_job_store.get_pending_view_generation(
                conn,
                status=CAPTURED_TEXT_VIEW_STATUS_PENDING,
                limit=limit,
            )

    def count(self) -> int:
        """저장된 captured text event 수를 반환한다."""

        with self._connect() as conn:
            return event_store.count_events(conn)

    def count_by_view_generation_status(self) -> dict[str, int]:
        """view generation 상태별 event 수를 반환한다."""

        with self._connect() as conn:
            return view_job_store.count_by_status(conn)

    def count_by_analysis_status(self) -> dict[str, int]:
        """분석 job 상태별 event 수를 반환한다."""

        with self._connect() as conn:
            return analysis_job_store.count_by_status(conn)

    def get_pending_analysis_sources(
        self,
        *,
        limit: int = 100,
    ) -> list[CapturedTextAnalysisSourceRecord]:
        """분석 대기 상태인 generated view를 오래된 순서로 반환한다."""

        _require_positive_limit(limit)
        with self._connect() as conn:
            return analysis_job_store.get_pending_sources(
                conn,
                ready_view_status=CAPTURED_TEXT_VIEW_STATUS_READY,
                pending_analysis_status=CAPTURED_TEXT_ANALYSIS_STATUS_PENDING,
                limit=limit,
            )

    def mark_analysis_completed(self, *, event_id: str, analysis_id: str) -> int:
        """단일 captured text event의 analysis job을 completed로 표시한다."""

        if not analysis_id.strip():
            raise ValueError("analysis_id must not be empty.")
        with self._connect() as conn:
            return analysis_job_store.mark_completed(
                conn,
                event_id=event_id,
                analysis_id=analysis_id,
            )

    def mark_analysis_failed(self, *, event_id: str, error_message: str) -> int:
        """단일 captured text event의 analysis job을 failed로 표시한다."""

        with self._connect() as conn:
            return analysis_job_store.mark_failed(
                conn,
                event_id=event_id,
                error_message=error_message,
            )

    def mark_view_generation_status(self, *, event_id: str, status: str) -> int:
        """단일 event의 view generation 상태를 갱신한다."""

        with self._connect() as conn:
            return view_job_store.update_status(
                conn,
                event_id=event_id,
                status=status,
            )

    def get_recent_view_generation_by_status(
        self,
        *,
        status: str,
        limit: int = 50,
    ) -> list[CapturedTextRecord]:
        """view generation job 상태로 최근 raw event를 반환한다."""

        _require_positive_limit(limit)
        with self._connect() as conn:
            return view_job_store.get_recent_by_status(
                conn,
                status=status,
                limit=limit,
            )

    def save_generated_view(self, record: CapturedTextGeneratedViewRecord) -> None:
        """weak/strong generated view를 저장한다."""

        with self._connect() as conn:
            generated_view_store.save_generated_view(conn, record)
            analysis_job_store.upsert_job(
                conn,
                event_id=record.event_id,
                status=CAPTURED_TEXT_ANALYSIS_STATUS_PENDING,
                metadata={
                    "source_text_fingerprint": record.source_text_fingerprint,
                    "generator_name": record.generator_name,
                    "generator_version": record.generator_version,
                },
            )

    def get_generated_view(
        self,
        event_id: str,
    ) -> CapturedTextGeneratedViewRecord | None:
        """event_id에 해당하는 generated view를 반환한다."""

        with self._connect() as conn:
            return generated_view_store.get_generated_view(conn, event_id)

    def delete_generated_view(self, event_id: str) -> int:
        """event_id에 해당하는 generated view를 삭제한다."""

        with self._connect() as conn:
            return generated_view_store.delete_generated_view(conn, event_id)

    def get_recent_generated_views(
        self,
        *,
        limit: int = 50,
    ) -> list[CapturedTextGeneratedViewRecord]:
        """최근 generated view를 최신순으로 반환한다."""

        _require_positive_limit(limit)
        with self._connect() as conn:
            return generated_view_store.get_recent_generated_views(conn, limit=limit)

    def count_generated_views(self) -> int:
        """저장된 generated view 수를 반환한다."""

        with self._connect() as conn:
            return generated_view_store.count_generated_views(conn)

    def get_ready_generated_training_sources(
        self,
        *,
        cutoff: datetime,
        limit: int,
    ) -> list[CapturedTextGeneratedTrainingSourceRecord]:
        """ready 상태 generated view를 학습 source 후보로 반환한다."""

        _require_positive_limit(limit)
        with self._connect() as conn:
            return generated_view_store.get_ready_generated_training_sources(
                conn,
                ready_status=CAPTURED_TEXT_VIEW_STATUS_READY,
                cutoff=cutoff,
                limit=limit,
            )

    def delete_older_than(self, *, cutoff: datetime) -> int:
        """cutoff보다 오래된 captured text event를 삭제한다."""

        with self._connect() as conn:
            return retention.delete_older_than(conn, cutoff=cutoff)

    def delete_oldest_excess(self, *, keep_latest: int) -> int:
        """최신 keep_latest개를 제외한 오래된 event를 삭제한다."""

        if keep_latest < 0:
            raise ValueError("keep_latest must not be negative.")
        with self._connect() as conn:
            return retention.delete_oldest_excess(conn, keep_latest=keep_latest)

    def _connect(self) -> sqlite3.Connection:
        return connect_agent_local_db(self.db_path)


def _require_positive_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive.")
