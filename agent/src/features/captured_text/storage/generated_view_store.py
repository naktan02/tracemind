"""Captured text generated view table store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from agent.src.features.captured_text.storage.records import (
    CapturedTextGeneratedTrainingSourceRecord,
    CapturedTextGeneratedViewRecord,
)

INSERT_GENERATED_VIEW_SQL = """
INSERT OR REPLACE INTO captured_text_generated_views
    (event_id, schema_version, generated_at, weak_text, strong_text_0,
     strong_text_1, generator_name, generator_version, source_text_fingerprint,
     metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

SELECT_GENERATED_VIEW_SQL = """
SELECT event_id, schema_version, generated_at, weak_text, strong_text_0,
       strong_text_1, generator_name, generator_version, source_text_fingerprint,
       metadata
FROM captured_text_generated_views
WHERE event_id = ?;
"""

DELETE_GENERATED_VIEW_SQL = """
DELETE FROM captured_text_generated_views
WHERE event_id = ?;
"""

SELECT_RECENT_GENERATED_VIEWS_SQL = """
SELECT event_id, schema_version, generated_at, weak_text, strong_text_0,
       strong_text_1, generator_name, generator_version, source_text_fingerprint,
       metadata
FROM captured_text_generated_views
ORDER BY generated_at DESC
LIMIT ?;
"""

COUNT_GENERATED_VIEWS_SQL = "SELECT COUNT(*) FROM captured_text_generated_views;"

SELECT_READY_GENERATED_TRAINING_SOURCES_SQL = """
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


def save_generated_view(
    conn: sqlite3.Connection,
    record: CapturedTextGeneratedViewRecord,
) -> None:
    conn.execute(
        INSERT_GENERATED_VIEW_SQL,
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


def get_generated_view(
    conn: sqlite3.Connection,
    event_id: str,
) -> CapturedTextGeneratedViewRecord | None:
    row = conn.execute(SELECT_GENERATED_VIEW_SQL, (event_id,)).fetchone()
    return None if row is None else row_to_generated_view(row)


def delete_generated_view(conn: sqlite3.Connection, event_id: str) -> int:
    cursor = conn.execute(DELETE_GENERATED_VIEW_SQL, (event_id,))
    return max(cursor.rowcount, 0)


def get_recent_generated_views(
    conn: sqlite3.Connection,
    *,
    limit: int,
) -> list[CapturedTextGeneratedViewRecord]:
    rows = conn.execute(SELECT_RECENT_GENERATED_VIEWS_SQL, (limit,)).fetchall()
    return [row_to_generated_view(row) for row in rows]


def count_generated_views(conn: sqlite3.Connection) -> int:
    return conn.execute(COUNT_GENERATED_VIEWS_SQL).fetchone()[0]


def get_ready_generated_training_sources(
    conn: sqlite3.Connection,
    *,
    ready_status: str,
    cutoff: datetime,
    limit: int,
) -> list[CapturedTextGeneratedTrainingSourceRecord]:
    rows = conn.execute(
        SELECT_READY_GENERATED_TRAINING_SOURCES_SQL,
        (ready_status, cutoff.isoformat(), limit),
    ).fetchall()
    return [row_to_generated_training_source(row) for row in rows]


def row_to_generated_view(row: tuple[Any, ...]) -> CapturedTextGeneratedViewRecord:
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


def row_to_generated_training_source(
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
