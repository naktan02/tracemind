"""Captured text SQLite schema ownership."""

from __future__ import annotations

import sqlite3

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


def reset_legacy_schema(conn: sqlite3.Connection) -> None:
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
