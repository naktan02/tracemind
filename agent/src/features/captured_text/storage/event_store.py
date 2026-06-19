"""Captured text raw event table store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from agent.src.features.captured_text.storage.records import (
    CapturedTextRecord,
    record_payload,
    text_fingerprint,
)

INSERT_SQL = """
INSERT OR REPLACE INTO captured_text_events
    (event_id, schema_version, occurred_at, received_at, text, locale,
     source_type, surface_type, page_url, page_title, collector_version,
     text_fingerprint, duplicate_of_event_id, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

SELECT_ONE_SQL = """
SELECT event_id, schema_version, occurred_at, received_at, text, locale,
       source_type, surface_type, page_url, page_title, collector_version,
       text_fingerprint, duplicate_of_event_id, metadata
FROM captured_text_events
WHERE event_id = ?;
"""

SELECT_RECENT_SQL = """
SELECT event_id, schema_version, occurred_at, received_at, text, locale,
       source_type, surface_type, page_url, page_title, collector_version,
       text_fingerprint, duplicate_of_event_id, metadata
FROM captured_text_events
ORDER BY occurred_at DESC
LIMIT ?;
"""

SELECT_ORIGINAL_BY_FINGERPRINT_SQL = """
SELECT event_id
FROM captured_text_events
WHERE text_fingerprint = ?
  AND event_id <> ?
  AND duplicate_of_event_id IS NULL
ORDER BY occurred_at ASC
LIMIT 1;
"""

COUNT_SQL = "SELECT COUNT(*) FROM captured_text_events;"


def save_event(
    conn: sqlite3.Connection,
    record: CapturedTextRecord,
) -> CapturedTextRecord:
    """raw captured text event를 저장하고 dedup 적용 record를 반환한다."""

    normalized = record_with_dedup_status(conn, record)
    conn.execute(
        INSERT_SQL,
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
    return normalized


def get_event(conn: sqlite3.Connection, event_id: str) -> CapturedTextRecord | None:
    row = conn.execute(SELECT_ONE_SQL, (event_id,)).fetchone()
    return None if row is None else row_to_record(row)


def get_recent_events(
    conn: sqlite3.Connection,
    *,
    limit: int,
) -> list[CapturedTextRecord]:
    rows = conn.execute(SELECT_RECENT_SQL, (limit,)).fetchall()
    return [row_to_record(row) for row in rows]


def count_events(conn: sqlite3.Connection) -> int:
    return conn.execute(COUNT_SQL).fetchone()[0]


def row_to_record(row: tuple[Any, ...]) -> CapturedTextRecord:
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
        text_fingerprint_value,
        duplicate_of_event_id,
        metadata_json,
    ) = row
    metadata = json.loads(str(metadata_json))
    if not isinstance(metadata, dict):
        metadata = {}
    return CapturedTextRecord(
        event_id=str(event_id),
        schema_version=str(schema_version),
        occurred_at=datetime_from_row(occurred_at),
        received_at=datetime_from_row(received_at),
        text=str(text),
        locale=str(locale),
        source_type=str(source_type),
        surface_type=str(surface_type),
        page_url=None if page_url is None else str(page_url),
        page_title=None if page_title is None else str(page_title),
        collector_version=None if collector_version is None else str(collector_version),
        text_fingerprint=str(text_fingerprint_value),
        duplicate_of_event_id=(
            None if duplicate_of_event_id is None else str(duplicate_of_event_id)
        ),
        metadata=metadata,
    )


def record_with_dedup_status(
    conn: sqlite3.Connection,
    record: CapturedTextRecord,
) -> CapturedTextRecord:
    fingerprint = record.text_fingerprint or text_fingerprint(
        text=record.text,
        locale=record.locale,
        source_type=record.source_type,
        surface_type=record.surface_type,
    )
    original = conn.execute(
        SELECT_ORIGINAL_BY_FINGERPRINT_SQL,
        (fingerprint, record.event_id),
    ).fetchone()
    if original is None:
        return CapturedTextRecord(
            **{
                **record_payload(record),
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
            **record_payload(record),
            "text_fingerprint": fingerprint,
            "duplicate_of_event_id": original_event_id,
            "metadata": metadata,
        }
    )


def datetime_from_row(value: object) -> datetime:
    return datetime.fromisoformat(str(value))
