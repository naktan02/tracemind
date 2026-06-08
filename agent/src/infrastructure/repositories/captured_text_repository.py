"""CapturedTextEvent agent-local SQLite 저장소."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from agent.src.contracts.captured_text_contracts import (
    CAPTURED_TEXT_EVENT_V1,
    CapturedTextEventPayload,
)

_DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "captured_text.db"
CAPTURED_TEXT_VIEW_STATUS_PENDING = "pending"
CAPTURED_TEXT_VIEW_STATUS_DUPLICATE = "duplicate"
CAPTURED_TEXT_VIEW_STATUS_READY = "ready"
CAPTURED_TEXT_VIEW_STATUS_FAILED = "failed"
CAPTURED_TEXT_ANALYSIS_STATUS_PENDING = "pending"
CAPTURED_TEXT_ANALYSIS_STATUS_COMPLETED = "completed"
CAPTURED_TEXT_ANALYSIS_STATUS_FAILED = "failed"
CAPTURED_TEXT_GENERATED_VIEW_V1 = "captured_text_generated_view.v1"

_CAPTURED_TEXT_VIEW_STATUSES = frozenset(
    {
        CAPTURED_TEXT_VIEW_STATUS_PENDING,
        CAPTURED_TEXT_VIEW_STATUS_DUPLICATE,
        CAPTURED_TEXT_VIEW_STATUS_READY,
        CAPTURED_TEXT_VIEW_STATUS_FAILED,
    }
)

_CAPTURED_TEXT_ANALYSIS_STATUSES = frozenset(
    {
        CAPTURED_TEXT_ANALYSIS_STATUS_PENDING,
        CAPTURED_TEXT_ANALYSIS_STATUS_COMPLETED,
        CAPTURED_TEXT_ANALYSIS_STATUS_FAILED,
    }
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS captured_text_events (
    event_id          TEXT PRIMARY KEY,
    schema_version    TEXT NOT NULL,
    occurred_at       TEXT NOT NULL,
    received_at       TEXT NOT NULL,
    text              TEXT NOT NULL,
    locale            TEXT NOT NULL,
    source_type       TEXT NOT NULL,
    surface_type      TEXT NOT NULL,
    page_url          TEXT,
    page_title        TEXT,
    collector_version TEXT,
    text_fingerprint  TEXT NOT NULL DEFAULT '',
    duplicate_of_event_id TEXT,
    metadata          TEXT NOT NULL
);
"""

_CREATE_VIEW_GENERATION_JOB_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS captured_text_view_generation_jobs (
    event_id      TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    error_message TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (event_id)
        REFERENCES captured_text_events (event_id)
        ON DELETE CASCADE
);
"""

_CREATE_GENERATED_VIEW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS captured_text_generated_views (
    event_id                TEXT PRIMARY KEY,
    schema_version          TEXT NOT NULL,
    generated_at            TEXT NOT NULL,
    weak_text               TEXT NOT NULL,
    strong_text_0           TEXT NOT NULL,
    strong_text_1           TEXT NOT NULL,
    generator_name          TEXT NOT NULL,
    generator_version       TEXT NOT NULL,
    source_text_fingerprint TEXT NOT NULL,
    metadata                TEXT NOT NULL,
    FOREIGN KEY (event_id)
        REFERENCES captured_text_events (event_id)
        ON DELETE CASCADE
);
"""

_CREATE_ANALYSIS_JOB_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS captured_text_analysis_jobs (
    event_id      TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    analysis_id   TEXT,
    error_message TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (event_id)
        REFERENCES captured_text_events (event_id)
        ON DELETE CASCADE
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO captured_text_events
    (event_id, schema_version, occurred_at, received_at, text, locale,
     source_type, surface_type, page_url, page_title, collector_version,
     text_fingerprint, duplicate_of_event_id, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_ONE_SQL = """
SELECT event_id, schema_version, occurred_at, received_at, text, locale,
       source_type, surface_type, page_url, page_title, collector_version,
       text_fingerprint, duplicate_of_event_id, metadata
FROM captured_text_events
WHERE event_id = ?;
"""

_SELECT_RECENT_SQL = """
SELECT event_id, schema_version, occurred_at, received_at, text, locale,
       source_type, surface_type, page_url, page_title, collector_version,
       text_fingerprint, duplicate_of_event_id, metadata
FROM captured_text_events
ORDER BY occurred_at DESC
LIMIT ?;
"""

_SELECT_PENDING_VIEW_GENERATION_SQL = """
SELECT e.event_id, e.schema_version, e.occurred_at, e.received_at, e.text, e.locale,
       e.source_type, e.surface_type, e.page_url, e.page_title, e.collector_version,
       e.text_fingerprint, e.duplicate_of_event_id, e.metadata
FROM captured_text_events e
JOIN captured_text_view_generation_jobs j ON e.event_id = j.event_id
WHERE j.status = ?
ORDER BY e.occurred_at ASC
LIMIT ?;
"""

_SELECT_RECENT_VIEW_GENERATION_BY_STATUS_SQL = """
SELECT e.event_id, e.schema_version, e.occurred_at, e.received_at, e.text, e.locale,
       e.source_type, e.surface_type, e.page_url, e.page_title, e.collector_version,
       e.text_fingerprint, e.duplicate_of_event_id, e.metadata
FROM captured_text_events e
JOIN captured_text_view_generation_jobs j ON e.event_id = j.event_id
WHERE j.status = ?
ORDER BY e.occurred_at DESC
LIMIT ?;
"""

_SELECT_ORIGINAL_BY_FINGERPRINT_SQL = """
SELECT event_id
FROM captured_text_events
WHERE text_fingerprint = ?
  AND event_id <> ?
  AND duplicate_of_event_id IS NULL
ORDER BY occurred_at ASC
LIMIT 1;
"""

_COUNT_SQL = "SELECT COUNT(*) FROM captured_text_events;"

_COUNT_BY_STATUS_SQL = """
SELECT status, COUNT(*)
FROM captured_text_view_generation_jobs
GROUP BY status;
"""

_UPDATE_VIEW_GENERATION_STATUS_SQL = """
UPDATE captured_text_view_generation_jobs
SET status = ?, updated_at = ?, error_message = NULL
WHERE event_id = ?;
"""

_UPSERT_VIEW_GENERATION_JOB_SQL = """
INSERT INTO captured_text_view_generation_jobs
    (event_id, status, updated_at, error_message, metadata)
VALUES (?, ?, ?, NULL, ?)
ON CONFLICT(event_id) DO UPDATE SET
    status = excluded.status,
    updated_at = excluded.updated_at,
    error_message = NULL,
    metadata = excluded.metadata;
"""

_DELETE_OLDER_THAN_SQL = """
DELETE FROM captured_text_events
WHERE occurred_at < ?;
"""

_DELETE_EXCESS_SQL = """
DELETE FROM captured_text_events
WHERE event_id IN (
    SELECT event_id
    FROM captured_text_events
    ORDER BY occurred_at DESC
    LIMIT -1 OFFSET ?
);
"""

_DELETE_ORPHANED_GENERATED_VIEWS_SQL = """
DELETE FROM captured_text_generated_views
WHERE event_id NOT IN (
    SELECT event_id FROM captured_text_events
);
"""

_DELETE_ORPHANED_VIEW_GENERATION_JOBS_SQL = """
DELETE FROM captured_text_view_generation_jobs
WHERE event_id NOT IN (
    SELECT event_id FROM captured_text_events
);
"""

_DELETE_ORPHANED_ANALYSIS_JOBS_SQL = """
DELETE FROM captured_text_analysis_jobs
WHERE event_id NOT IN (
    SELECT event_id FROM captured_text_events
);
"""

_INSERT_GENERATED_VIEW_SQL = """
INSERT OR REPLACE INTO captured_text_generated_views
    (event_id, schema_version, generated_at, weak_text, strong_text_0,
     strong_text_1, generator_name, generator_version, source_text_fingerprint,
     metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_GENERATED_VIEW_SQL = """
SELECT event_id, schema_version, generated_at, weak_text, strong_text_0,
       strong_text_1, generator_name, generator_version, source_text_fingerprint,
       metadata
FROM captured_text_generated_views
WHERE event_id = ?;
"""

_DELETE_GENERATED_VIEW_SQL = """
DELETE FROM captured_text_generated_views
WHERE event_id = ?;
"""

_SELECT_RECENT_GENERATED_VIEWS_SQL = """
SELECT event_id, schema_version, generated_at, weak_text, strong_text_0,
       strong_text_1, generator_name, generator_version, source_text_fingerprint,
       metadata
FROM captured_text_generated_views
ORDER BY generated_at DESC
LIMIT ?;
"""

_COUNT_GENERATED_VIEWS_SQL = "SELECT COUNT(*) FROM captured_text_generated_views;"

_COUNT_BY_ANALYSIS_STATUS_SQL = """
SELECT status, COUNT(*)
FROM captured_text_analysis_jobs
GROUP BY status;
"""

_UPSERT_ANALYSIS_JOB_SQL = """
INSERT INTO captured_text_analysis_jobs
    (event_id, status, updated_at, analysis_id, error_message, metadata)
VALUES (?, ?, ?, NULL, NULL, ?)
ON CONFLICT(event_id) DO UPDATE SET
    status = excluded.status,
    updated_at = excluded.updated_at,
    analysis_id = NULL,
    error_message = NULL,
    metadata = excluded.metadata;
"""

_SELECT_READY_GENERATED_TRAINING_SOURCES_SQL = """
SELECT e.event_id, e.occurred_at, e.text, e.locale, e.source_type, e.surface_type,
       e.text_fingerprint, v.generated_at, v.weak_text, v.strong_text_0,
       v.strong_text_1, v.generator_name, v.generator_version, v.metadata
FROM captured_text_events e
JOIN captured_text_view_generation_jobs j ON e.event_id = j.event_id
JOIN captured_text_generated_views v ON e.event_id = v.event_id
WHERE j.status = ?
  AND e.occurred_at >= ?
ORDER BY e.occurred_at DESC
LIMIT ?;
"""


@dataclass(slots=True)
class CapturedTextRecord:
    """agent 로컬 captured text raw event snapshot."""

    event_id: str
    occurred_at: datetime
    received_at: datetime
    text: str
    locale: str
    source_type: str
    surface_type: str
    page_url: str | None = None
    page_title: str | None = None
    collector_version: str | None = None
    text_fingerprint: str = ""
    duplicate_of_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CAPTURED_TEXT_EVENT_V1

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")
        if not self.text.strip():
            raise ValueError("text must not be empty.")
        if not self.locale.strip():
            raise ValueError("locale must not be empty.")
        if not self.source_type.strip():
            raise ValueError("source_type must not be empty.")
        if not self.surface_type.strip():
            raise ValueError("surface_type must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty.")
        if self.duplicate_of_event_id is not None and not (
            self.duplicate_of_event_id.strip()
        ):
            raise ValueError("duplicate_of_event_id must not be empty.")


@dataclass(slots=True)
class CapturedTextGeneratedViewRecord:
    """agent-local captured text에서 만든 weak/strong view snapshot."""

    event_id: str
    generated_at: datetime
    weak_text: str
    strong_text_0: str
    strong_text_1: str
    generator_name: str
    generator_version: str
    source_text_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CAPTURED_TEXT_GENERATED_VIEW_V1

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")
        if not self.weak_text.strip():
            raise ValueError("weak_text must not be empty.")
        if not self.strong_text_0.strip():
            raise ValueError("strong_text_0 must not be empty.")
        if not self.strong_text_1.strip():
            raise ValueError("strong_text_1 must not be empty.")
        if not self.generator_name.strip():
            raise ValueError("generator_name must not be empty.")
        if not self.generator_version.strip():
            raise ValueError("generator_version must not be empty.")
        if not self.source_text_fingerprint.strip():
            raise ValueError("source_text_fingerprint must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty.")


@dataclass(slots=True)
class CapturedTextGeneratedTrainingSourceRecord:
    """generated view와 원본 captured text를 합친 학습 source snapshot."""

    event_id: str
    occurred_at: datetime
    text: str
    locale: str
    source_type: str
    surface_type: str
    text_fingerprint: str
    generated_at: datetime
    weak_text: str
    strong_text_0: str
    strong_text_1: str
    generator_name: str
    generator_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


def captured_text_record_from_payload(
    payload: CapturedTextEventPayload,
    *,
    received_at: datetime | None = None,
) -> CapturedTextRecord:
    """CapturedTextEventPayload를 저장소 record로 정규화한다."""

    return CapturedTextRecord(
        event_id=payload.event_id,
        schema_version=payload.schema_version,
        occurred_at=payload.occurred_at,
        received_at=received_at or datetime.now(tz=timezone.utc),
        text=payload.text,
        locale=payload.locale,
        source_type=payload.source_type.value,
        surface_type=payload.surface_type.value,
        page_url=payload.page_url,
        page_title=payload.page_title,
        collector_version=payload.collector_version,
        text_fingerprint=_text_fingerprint(
            text=payload.text,
            locale=payload.locale,
            source_type=payload.source_type.value,
            surface_type=payload.surface_type.value,
        ),
        metadata=dict(payload.metadata),
    )


@dataclass(slots=True)
class CapturedTextRepository:
    """CapturedTextRecord를 SQLite에 저장하고 최소 조회를 제공한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            _reset_legacy_schema(conn)
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_VIEW_GENERATION_JOB_TABLE_SQL)
            conn.execute(_CREATE_GENERATED_VIEW_TABLE_SQL)
            conn.execute(_CREATE_ANALYSIS_JOB_TABLE_SQL)

    def save(self, record: CapturedTextRecord) -> None:
        """event_id 기준으로 raw captured text event를 저장한다."""

        with self._connect() as conn:
            normalized = _record_with_dedup_status(conn, record)
            conn.execute(
                _INSERT_SQL,
                (
                    normalized.event_id,
                    normalized.schema_version,
                    normalized.occurred_at.isoformat(),
                    normalized.received_at.isoformat(),
                    normalized.text,
                    normalized.locale,
                    normalized.source_type,
                    normalized.surface_type,
                    normalized.page_url,
                    normalized.page_title,
                    normalized.collector_version,
                    normalized.text_fingerprint,
                    normalized.duplicate_of_event_id,
                    json.dumps(normalized.metadata, ensure_ascii=False),
                ),
            )
            _upsert_view_generation_job(
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
            row = conn.execute(_SELECT_ONE_SQL, (event_id,)).fetchone()
        return None if row is None else _row_to_record(row)

    def get_recent(self, *, limit: int = 50) -> list[CapturedTextRecord]:
        """최근 captured text event를 최신순으로 반환한다."""

        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (limit,)).fetchall()
        return [_row_to_record(row) for row in rows]

    def get_pending_view_generation(
        self,
        *,
        limit: int = 100,
    ) -> list[CapturedTextRecord]:
        """view generation을 기다리는 raw event를 오래된 순서로 반환한다."""

        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(
                _SELECT_PENDING_VIEW_GENERATION_SQL,
                (CAPTURED_TEXT_VIEW_STATUS_PENDING, limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def count(self) -> int:
        """저장된 captured text event 수를 반환한다."""

        with self._connect() as conn:
            return conn.execute(_COUNT_SQL).fetchone()[0]

    def count_by_view_generation_status(self) -> dict[str, int]:
        """view generation 상태별 event 수를 반환한다."""

        with self._connect() as conn:
            rows = conn.execute(_COUNT_BY_STATUS_SQL).fetchall()
        return {str(status): int(count) for status, count in rows}

    def count_by_analysis_status(self) -> dict[str, int]:
        """분석 job 상태별 event 수를 반환한다."""

        with self._connect() as conn:
            rows = conn.execute(_COUNT_BY_ANALYSIS_STATUS_SQL).fetchall()
        return {str(status): int(count) for status, count in rows}

    def mark_view_generation_status(self, *, event_id: str, status: str) -> int:
        """단일 event의 view generation 상태를 갱신한다."""

        if status not in _CAPTURED_TEXT_VIEW_STATUSES:
            raise ValueError("status is unsupported.")
        with self._connect() as conn:
            cursor = conn.execute(
                _UPDATE_VIEW_GENERATION_STATUS_SQL,
                (status, datetime.now(tz=timezone.utc).isoformat(), event_id),
            )
        return max(cursor.rowcount, 0)

    def get_recent_view_generation_by_status(
        self,
        *,
        status: str,
        limit: int = 50,
    ) -> list[CapturedTextRecord]:
        """view generation job 상태로 최근 raw event를 반환한다."""

        if status not in _CAPTURED_TEXT_VIEW_STATUSES:
            raise ValueError("status is unsupported.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(
                _SELECT_RECENT_VIEW_GENERATION_BY_STATUS_SQL,
                (status, limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def save_generated_view(self, record: CapturedTextGeneratedViewRecord) -> None:
        """weak/strong generated view를 저장한다."""

        with self._connect() as conn:
            conn.execute(
                _INSERT_GENERATED_VIEW_SQL,
                (
                    record.event_id,
                    record.schema_version,
                    record.generated_at.isoformat(),
                    record.weak_text,
                    record.strong_text_0,
                    record.strong_text_1,
                    record.generator_name,
                    record.generator_version,
                    record.source_text_fingerprint,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
            _upsert_analysis_job(
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
            row = conn.execute(_SELECT_GENERATED_VIEW_SQL, (event_id,)).fetchone()
        return None if row is None else _row_to_generated_view(row)

    def delete_generated_view(self, event_id: str) -> int:
        """event_id에 해당하는 generated view를 삭제한다."""

        with self._connect() as conn:
            cursor = conn.execute(_DELETE_GENERATED_VIEW_SQL, (event_id,))
        return max(cursor.rowcount, 0)

    def get_recent_generated_views(
        self,
        *,
        limit: int = 50,
    ) -> list[CapturedTextGeneratedViewRecord]:
        """최근 generated view를 최신순으로 반환한다."""

        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_GENERATED_VIEWS_SQL, (limit,)).fetchall()
        return [_row_to_generated_view(row) for row in rows]

    def count_generated_views(self) -> int:
        """저장된 generated view 수를 반환한다."""

        with self._connect() as conn:
            return conn.execute(_COUNT_GENERATED_VIEWS_SQL).fetchone()[0]

    def get_ready_generated_training_sources(
        self,
        *,
        cutoff: datetime,
        limit: int,
    ) -> list[CapturedTextGeneratedTrainingSourceRecord]:
        """ready 상태 generated view를 학습 source 후보로 반환한다."""

        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(
                _SELECT_READY_GENERATED_TRAINING_SOURCES_SQL,
                (
                    CAPTURED_TEXT_VIEW_STATUS_READY,
                    cutoff.isoformat(),
                    limit,
                ),
            ).fetchall()
        return [_row_to_generated_training_source(row) for row in rows]

    def delete_older_than(self, *, cutoff: datetime) -> int:
        """cutoff보다 오래된 captured text event를 삭제한다."""

        with self._connect() as conn:
            cursor = conn.execute(_DELETE_OLDER_THAN_SQL, (cutoff.isoformat(),))
            conn.execute(_DELETE_ORPHANED_GENERATED_VIEWS_SQL)
            conn.execute(_DELETE_ORPHANED_VIEW_GENERATION_JOBS_SQL)
            conn.execute(_DELETE_ORPHANED_ANALYSIS_JOBS_SQL)
        return max(cursor.rowcount, 0)

    def delete_oldest_excess(self, *, keep_latest: int) -> int:
        """최신 keep_latest개를 제외한 오래된 event를 삭제한다."""

        if keep_latest < 0:
            raise ValueError("keep_latest must not be negative.")
        with self._connect() as conn:
            cursor = conn.execute(_DELETE_EXCESS_SQL, (keep_latest,))
            conn.execute(_DELETE_ORPHANED_GENERATED_VIEWS_SQL)
            conn.execute(_DELETE_ORPHANED_VIEW_GENERATION_JOBS_SQL)
            conn.execute(_DELETE_ORPHANED_ANALYSIS_JOBS_SQL)
        return max(cursor.rowcount, 0)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _row_to_record(row: tuple[Any, ...]) -> CapturedTextRecord:
    (
        event_id,
        schema_version,
        occurred_at,
        received_at,
        text,
        locale,
        source_type,
        surface_type,
        page_url,
        page_title,
        collector_version,
        text_fingerprint,
        duplicate_of_event_id,
        metadata_json,
    ) = row
    metadata = json.loads(str(metadata_json))
    if not isinstance(metadata, dict):
        metadata = {}
    return CapturedTextRecord(
        event_id=str(event_id),
        schema_version=str(schema_version),
        occurred_at=datetime.fromisoformat(str(occurred_at)),
        received_at=datetime.fromisoformat(str(received_at)),
        text=str(text),
        locale=str(locale),
        source_type=str(source_type),
        surface_type=str(surface_type),
        page_url=None if page_url is None else str(page_url),
        page_title=None if page_title is None else str(page_title),
        collector_version=None if collector_version is None else str(collector_version),
        text_fingerprint=str(text_fingerprint),
        duplicate_of_event_id=(
            None if duplicate_of_event_id is None else str(duplicate_of_event_id)
        ),
        metadata=metadata,
    )


def _row_to_generated_view(row: tuple[Any, ...]) -> CapturedTextGeneratedViewRecord:
    (
        event_id,
        schema_version,
        generated_at,
        weak_text,
        strong_text_0,
        strong_text_1,
        generator_name,
        generator_version,
        source_text_fingerprint,
        metadata_json,
    ) = row
    metadata = json.loads(str(metadata_json))
    if not isinstance(metadata, dict):
        metadata = {}
    return CapturedTextGeneratedViewRecord(
        event_id=str(event_id),
        schema_version=str(schema_version),
        generated_at=datetime.fromisoformat(str(generated_at)),
        weak_text=str(weak_text),
        strong_text_0=str(strong_text_0),
        strong_text_1=str(strong_text_1),
        generator_name=str(generator_name),
        generator_version=str(generator_version),
        source_text_fingerprint=str(source_text_fingerprint),
        metadata=metadata,
    )


def _row_to_generated_training_source(
    row: tuple[Any, ...],
) -> CapturedTextGeneratedTrainingSourceRecord:
    (
        event_id,
        occurred_at,
        text,
        locale,
        source_type,
        surface_type,
        text_fingerprint,
        generated_at,
        weak_text,
        strong_text_0,
        strong_text_1,
        generator_name,
        generator_version,
        metadata_json,
    ) = row
    metadata = json.loads(str(metadata_json))
    if not isinstance(metadata, dict):
        metadata = {}
    return CapturedTextGeneratedTrainingSourceRecord(
        event_id=str(event_id),
        occurred_at=datetime.fromisoformat(str(occurred_at)),
        text=str(text),
        locale=str(locale),
        source_type=str(source_type),
        surface_type=str(surface_type),
        text_fingerprint=str(text_fingerprint),
        generated_at=datetime.fromisoformat(str(generated_at)),
        weak_text=str(weak_text),
        strong_text_0=str(strong_text_0),
        strong_text_1=str(strong_text_1),
        generator_name=str(generator_name),
        generator_version=str(generator_version),
        metadata=metadata,
    )


def _record_with_dedup_status(
    conn: sqlite3.Connection,
    record: CapturedTextRecord,
) -> CapturedTextRecord:
    fingerprint = record.text_fingerprint or _text_fingerprint(
        text=record.text,
        locale=record.locale,
        source_type=record.source_type,
        surface_type=record.surface_type,
    )
    original = conn.execute(
        _SELECT_ORIGINAL_BY_FINGERPRINT_SQL,
        (fingerprint, record.event_id),
    ).fetchone()
    if original is None:
        return CapturedTextRecord(
            **{
                **_record_payload(record),
                "text_fingerprint": fingerprint,
                "duplicate_of_event_id": record.duplicate_of_event_id,
            }
        )
    original_event_id = str(original[0])
    metadata = {
        **record.metadata,
        "dedup_fingerprint": fingerprint,
        "duplicate_of_event_id": original_event_id,
    }
    return CapturedTextRecord(
        **{
            **_record_payload(record),
            "text_fingerprint": fingerprint,
            "duplicate_of_event_id": original_event_id,
            "metadata": metadata,
        }
    )


def _record_payload(record: CapturedTextRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "schema_version": record.schema_version,
        "occurred_at": record.occurred_at,
        "received_at": record.received_at,
        "text": record.text,
        "locale": record.locale,
        "source_type": record.source_type,
        "surface_type": record.surface_type,
        "page_url": record.page_url,
        "page_title": record.page_title,
        "collector_version": record.collector_version,
        "metadata": dict(record.metadata),
    }


def _text_fingerprint(
    *,
    text: str,
    locale: str,
    source_type: str,
    surface_type: str,
) -> str:
    payload = {
        "locale": locale.strip().lower(),
        "source_type": source_type.strip().lower(),
        "surface_type": surface_type.strip().lower(),
        "text": " ".join(text.strip().lower().split()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _upsert_view_generation_job(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    status: str,
    metadata: dict[str, object],
) -> None:
    if status not in _CAPTURED_TEXT_VIEW_STATUSES:
        raise ValueError("status is unsupported.")
    conn.execute(
        _UPSERT_VIEW_GENERATION_JOB_SQL,
        (
            event_id,
            status,
            datetime.now(tz=timezone.utc).isoformat(),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )


def _upsert_analysis_job(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    status: str,
    metadata: dict[str, object],
) -> None:
    if status not in _CAPTURED_TEXT_ANALYSIS_STATUSES:
        raise ValueError("status is unsupported.")
    conn.execute(
        _UPSERT_ANALYSIS_JOB_SQL,
        (
            event_id,
            status,
            datetime.now(tz=timezone.utc).isoformat(),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )


def _reset_legacy_schema(conn: sqlite3.Connection) -> None:
    """테스트 단계 destructive migration: legacy captured-text schema를 버린다."""

    table_names = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "captured_text_events" not in table_names:
        return

    event_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(captured_text_events)").fetchall()
    }
    legacy_event_columns = {
        "view_generation_status",
    }
    required_tables = {
        "captured_text_view_generation_jobs",
        "captured_text_generated_views",
        "captured_text_analysis_jobs",
    }
    if event_columns.isdisjoint(legacy_event_columns) and required_tables.issubset(
        table_names
    ):
        return

    conn.execute("DROP TABLE IF EXISTS captured_text_analysis_jobs")
    conn.execute("DROP TABLE IF EXISTS captured_text_generated_views")
    conn.execute("DROP TABLE IF EXISTS captured_text_view_generation_jobs")
    conn.execute("DROP TABLE IF EXISTS captured_text_events")
