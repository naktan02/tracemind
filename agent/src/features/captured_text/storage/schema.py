"""Captured text SQLite schema ownership."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from agent.src.features.captured_text.storage.records import (
    CAPTURED_TEXT_VIEW_STATUS_DUPLICATE,
    CAPTURED_TEXT_VIEW_STATUS_PENDING,
    CAPTURED_TEXT_VIEW_STATUSES,
    text_fingerprint,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    ensure_analysis_event_schema,
)

CREATE_EVENT_TABLE_SQL = """
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

CREATE_VIEW_GENERATION_JOB_TABLE_SQL = """
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

CREATE_GENERATED_VIEW_TABLE_SQL = """
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

CREATE_ANALYSIS_JOB_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS captured_text_analysis_jobs (
    event_id      TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    analysis_id   TEXT,
    error_message TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (event_id)
        REFERENCES captured_text_events (event_id)
        ON DELETE CASCADE,
    FOREIGN KEY (analysis_id)
        REFERENCES analysis_events (analysis_id)
        ON DELETE SET NULL
);
"""


def ensure_captured_text_schema(conn: sqlite3.Connection) -> None:
    """Captured text schema와 의존 analysis schema를 준비한다."""

    reset_legacy_schema(conn)
    ensure_analysis_event_schema(conn)
    conn.execute(CREATE_EVENT_TABLE_SQL)
    conn.execute(CREATE_VIEW_GENERATION_JOB_TABLE_SQL)
    conn.execute(CREATE_GENERATED_VIEW_TABLE_SQL)
    conn.execute(CREATE_ANALYSIS_JOB_TABLE_SQL)
    repair_analysis_job_legacy_fk(conn)


def reset_legacy_schema(conn: sqlite3.Connection) -> None:
    """legacy captured-text schema를 현재 raw event/job table로 보존 migration한다."""

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

    legacy_events = _read_legacy_event_snapshots(conn)
    conn.execute("DROP TABLE IF EXISTS captured_text_analysis_jobs")
    conn.execute("DROP TABLE IF EXISTS captured_text_generated_views")
    conn.execute("DROP TABLE IF EXISTS captured_text_view_generation_jobs")
    conn.execute("DROP TABLE IF EXISTS captured_text_events")
    conn.execute(CREATE_EVENT_TABLE_SQL)
    conn.execute(CREATE_VIEW_GENERATION_JOB_TABLE_SQL)
    conn.executemany(
        """
        INSERT OR REPLACE INTO captured_text_events
            (event_id, schema_version, occurred_at, received_at, text, locale,
             source_type, surface_type, page_url, page_title, collector_version,
             text_fingerprint, duplicate_of_event_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (event_row for event_row, _job_row in legacy_events),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO captured_text_view_generation_jobs
            (event_id, status, updated_at, error_message, metadata)
        VALUES (?, ?, ?, ?, ?);
        """,
        (job_row for _event_row, job_row in legacy_events),
    )


def _read_legacy_event_snapshots(
    conn: sqlite3.Connection,
) -> list[tuple[tuple[object, ...], tuple[object, ...]]]:
    columns = [
        str(row[1])
        for row in conn.execute("PRAGMA table_info(captured_text_events)").fetchall()
    ]
    rows = conn.execute("SELECT * FROM captured_text_events").fetchall()
    return [
        migrated
        for row in rows
        if (migrated := _legacy_event_snapshot(dict(zip(columns, row)))) is not None
    ]


def _legacy_event_snapshot(
    row: dict[str, object],
) -> tuple[tuple[object, ...], tuple[object, ...]] | None:
    event_id = _clean_text(row.get("event_id"))
    text = _clean_text(row.get("text"))
    if event_id is None or text is None:
        return None

    now = datetime.now(tz=timezone.utc).isoformat()
    occurred_at = _clean_text(row.get("occurred_at")) or now
    received_at = _clean_text(row.get("received_at")) or occurred_at
    locale = _clean_text(row.get("locale")) or "unknown"
    source_type = _clean_text(row.get("source_type")) or "unknown"
    surface_type = _clean_text(row.get("surface_type")) or "unknown"
    duplicate_of_event_id = _clean_text(row.get("duplicate_of_event_id"))
    fingerprint = _clean_text(row.get("text_fingerprint")) or text_fingerprint(
        text=text,
        locale=locale,
        source_type=source_type,
        surface_type=surface_type,
    )
    metadata = _metadata_json(row.get("metadata"))
    status = _view_generation_status(row, duplicate_of_event_id=duplicate_of_event_id)

    event_row = (
        event_id,
        _clean_text(row.get("schema_version")) or "captured_text_event.v1",
        occurred_at,
        received_at,
        text,
        locale,
        source_type,
        surface_type,
        _clean_text(row.get("page_url")),
        _clean_text(row.get("page_title")),
        _clean_text(row.get("collector_version")),
        fingerprint,
        duplicate_of_event_id,
        metadata,
    )
    job_row = (
        event_id,
        status,
        received_at,
        _clean_text(row.get("view_generation_error"))
        or _clean_text(row.get("error_message")),
        _job_metadata_json(duplicate_of_event_id=duplicate_of_event_id),
    )
    return event_row, job_row


def _view_generation_status(
    row: dict[str, object],
    *,
    duplicate_of_event_id: str | None,
) -> str:
    status = _clean_text(row.get("view_generation_status"))
    if status in CAPTURED_TEXT_VIEW_STATUSES:
        return status
    if duplicate_of_event_id is not None:
        return CAPTURED_TEXT_VIEW_STATUS_DUPLICATE
    return CAPTURED_TEXT_VIEW_STATUS_PENDING


def _metadata_json(value: object) -> str:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return "{}"
        return json.dumps(
            parsed if isinstance(parsed, dict) else {},
            ensure_ascii=False,
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "{}"


def _job_metadata_json(*, duplicate_of_event_id: str | None) -> str:
    if duplicate_of_event_id is None:
        return "{}"
    return json.dumps(
        {"duplicate_of_event_id": duplicate_of_event_id},
        ensure_ascii=False,
    )


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def repair_analysis_job_legacy_fk(conn: sqlite3.Connection) -> None:
    """analysis_events_legacy를 바라보는 깨진 FK를 현재 analysis_events로 복구한다."""

    table_names = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "captured_text_analysis_jobs" not in table_names:
        return

    fk_parent_tables = {
        str(row[2])
        for row in conn.execute(
            "PRAGMA foreign_key_list(captured_text_analysis_jobs)"
        ).fetchall()
    }
    if "analysis_events_legacy" not in fk_parent_tables:
        return

    rows = conn.execute(
        """
        SELECT event_id, status, updated_at, analysis_id, error_message, metadata
        FROM captured_text_analysis_jobs;
        """
    ).fetchall()
    conn.execute("DROP TABLE captured_text_analysis_jobs")
    conn.execute(CREATE_ANALYSIS_JOB_TABLE_SQL)
    conn.executemany(
        """
        INSERT INTO captured_text_analysis_jobs
            (event_id, status, updated_at, analysis_id, error_message, metadata)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            (
                event_id,
                status,
                updated_at,
                _valid_analysis_id(conn, analysis_id),
                error_message,
                metadata,
            )
            for (
                event_id,
                status,
                updated_at,
                analysis_id,
                error_message,
                metadata,
            ) in rows
        ),
    )


def _valid_analysis_id(conn: sqlite3.Connection, analysis_id: object) -> object | None:
    if analysis_id is None:
        return None
    row = conn.execute(
        "SELECT 1 FROM analysis_events WHERE analysis_id = ? LIMIT 1;",
        (analysis_id,),
    ).fetchone()
    return analysis_id if row is not None else None
