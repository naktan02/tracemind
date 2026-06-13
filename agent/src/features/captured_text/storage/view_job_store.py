"""Captured text view generation job table store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from agent.src.features.captured_text.storage.event_store import (
    row_to_record,
)
from agent.src.features.captured_text.storage.records import (
    CAPTURED_TEXT_VIEW_STATUSES,
    CapturedTextRecord,
)

SELECT_PENDING_VIEW_GENERATION_SQL = """
SELECT e.event_id, e.schema_version, e.occurred_at, e.received_at, e.text, e.locale,
       e.source_type, e.surface_type, e.page_url, e.page_title, e.collector_version,
       e.text_fingerprint, e.duplicate_of_event_id, e.metadata
FROM captured_text_events e
JOIN captured_text_view_generation_jobs j ON e.event_id = j.event_id
WHERE j.status = ?
ORDER BY e.occurred_at ASC
LIMIT ?;
"""

SELECT_RECENT_VIEW_GENERATION_BY_STATUS_SQL = """
SELECT e.event_id, e.schema_version, e.occurred_at, e.received_at, e.text, e.locale,
       e.source_type, e.surface_type, e.page_url, e.page_title, e.collector_version,
       e.text_fingerprint, e.duplicate_of_event_id, e.metadata
FROM captured_text_events e
JOIN captured_text_view_generation_jobs j ON e.event_id = j.event_id
WHERE j.status = ?
ORDER BY e.occurred_at DESC
LIMIT ?;
"""

COUNT_BY_STATUS_SQL = """
SELECT status, COUNT(*)
FROM captured_text_view_generation_jobs
GROUP BY status;
"""

UPDATE_VIEW_GENERATION_STATUS_SQL = """
UPDATE captured_text_view_generation_jobs
SET status = ?, updated_at = ?, error_message = NULL
WHERE event_id = ?;
"""

UPSERT_VIEW_GENERATION_JOB_SQL = """
INSERT INTO captured_text_view_generation_jobs
    (event_id, status, updated_at, error_message, metadata)
VALUES (?, ?, ?, NULL, ?)
ON CONFLICT(event_id) DO UPDATE SET
    status = excluded.status,
    updated_at = excluded.updated_at,
    error_message = NULL,
    metadata = excluded.metadata;
"""


def get_pending_view_generation(
    conn: sqlite3.Connection,
    *,
    status: str,
    limit: int,
) -> list[CapturedTextRecord]:
    rows = conn.execute(SELECT_PENDING_VIEW_GENERATION_SQL, (status, limit)).fetchall()
    return [row_to_record(row) for row in rows]


def get_recent_by_status(
    conn: sqlite3.Connection,
    *,
    status: str,
    limit: int,
) -> list[CapturedTextRecord]:
    validate_status(status)
    rows = conn.execute(
        SELECT_RECENT_VIEW_GENERATION_BY_STATUS_SQL,
        (status, limit),
    ).fetchall()
    return [row_to_record(row) for row in rows]


def count_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(COUNT_BY_STATUS_SQL).fetchall()
    return {str(status): int(count) for status, count in rows}


def update_status(conn: sqlite3.Connection, *, event_id: str, status: str) -> int:
    validate_status(status)
    cursor = conn.execute(
        UPDATE_VIEW_GENERATION_STATUS_SQL,
        (status, datetime.now(tz=timezone.utc).isoformat(), event_id),
    )
    return max(cursor.rowcount, 0)


def upsert_job(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    status: str,
    metadata: dict[str, object],
) -> None:
    validate_status(status)
    conn.execute(
        UPSERT_VIEW_GENERATION_JOB_SQL,
        (
            event_id,
            status,
            datetime.now(tz=timezone.utc).isoformat(),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )


def validate_status(status: str) -> None:
    if status not in CAPTURED_TEXT_VIEW_STATUSES:
        raise ValueError("status is unsupported.")
