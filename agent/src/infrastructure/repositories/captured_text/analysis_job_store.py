"""Captured text analysis job table store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from agent.src.infrastructure.repositories.captured_text.records import (
    CAPTURED_TEXT_ANALYSIS_STATUS_COMPLETED,
    CAPTURED_TEXT_ANALYSIS_STATUS_FAILED,
    CAPTURED_TEXT_ANALYSIS_STATUSES,
    CapturedTextAnalysisSourceRecord,
)

COUNT_BY_ANALYSIS_STATUS_SQL = """
SELECT status, COUNT(*)
FROM captured_text_analysis_jobs
GROUP BY status;
"""

UPSERT_ANALYSIS_JOB_SQL = """
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

SELECT_PENDING_ANALYSIS_SOURCES_SQL = """
SELECT e.event_id, e.occurred_at, e.text, e.locale, e.source_type, e.surface_type,
       e.text_fingerprint, v.generated_at, v.weak_text, v.strong_text_0,
       v.strong_text_1, v.generator_name, v.generator_version, v.metadata
FROM captured_text_events e
JOIN captured_text_view_generation_jobs vj ON e.event_id = vj.event_id
JOIN captured_text_generated_views v ON e.event_id = v.event_id
JOIN captured_text_analysis_jobs aj ON e.event_id = aj.event_id
WHERE vj.status = ?
  AND aj.status = ?
  AND e.duplicate_of_event_id IS NULL
ORDER BY e.occurred_at ASC
LIMIT ?;
"""

UPDATE_ANALYSIS_JOB_COMPLETED_SQL = """
UPDATE captured_text_analysis_jobs
SET status = ?, updated_at = ?, analysis_id = ?, error_message = NULL
WHERE event_id = ?;
"""

UPDATE_ANALYSIS_JOB_FAILED_SQL = """
UPDATE captured_text_analysis_jobs
SET status = ?, updated_at = ?, error_message = ?
WHERE event_id = ?;
"""


def count_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(COUNT_BY_ANALYSIS_STATUS_SQL).fetchall()
    return {str(status): int(count) for status, count in rows}


def upsert_job(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    status: str,
    metadata: dict[str, object],
) -> None:
    validate_status(status)
    conn.execute(
        UPSERT_ANALYSIS_JOB_SQL,
        (
            event_id,
            status,
            datetime.now(tz=timezone.utc).isoformat(),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )


def get_pending_sources(
    conn: sqlite3.Connection,
    *,
    ready_view_status: str,
    pending_analysis_status: str,
    limit: int,
) -> list[CapturedTextAnalysisSourceRecord]:
    rows = conn.execute(
        SELECT_PENDING_ANALYSIS_SOURCES_SQL,
        (ready_view_status, pending_analysis_status, limit),
    ).fetchall()
    return [row_to_analysis_source(row) for row in rows]


def mark_completed(conn: sqlite3.Connection, *, event_id: str, analysis_id: str) -> int:
    cursor = conn.execute(
        UPDATE_ANALYSIS_JOB_COMPLETED_SQL,
        (
            CAPTURED_TEXT_ANALYSIS_STATUS_COMPLETED,
            datetime.now(tz=timezone.utc).isoformat(),
            analysis_id,
            event_id,
        ),
    )
    return max(cursor.rowcount, 0)


def mark_failed(conn: sqlite3.Connection, *, event_id: str, error_message: str) -> int:
    cursor = conn.execute(
        UPDATE_ANALYSIS_JOB_FAILED_SQL,
        (
            CAPTURED_TEXT_ANALYSIS_STATUS_FAILED,
            datetime.now(tz=timezone.utc).isoformat(),
            error_message,
            event_id,
        ),
    )
    return max(cursor.rowcount, 0)


def row_to_analysis_source(row: tuple[Any, ...]) -> CapturedTextAnalysisSourceRecord:
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
    return CapturedTextAnalysisSourceRecord(
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


def validate_status(status: str) -> None:
    if status not in CAPTURED_TEXT_ANALYSIS_STATUSES:
        raise ValueError("status is unsupported.")
