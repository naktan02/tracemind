"""Captured text retention SQL."""

from __future__ import annotations

import sqlite3
from datetime import datetime

DELETE_OLDER_THAN_SQL = """
DELETE FROM captured_text_events
WHERE occurred_at < ?;
"""

DELETE_EXCESS_SQL = """
DELETE FROM captured_text_events
WHERE event_id IN (
    SELECT event_id
    FROM captured_text_events
    ORDER BY occurred_at DESC
    LIMIT -1 OFFSET ?
);
"""

DELETE_ORPHANED_GENERATED_VIEWS_SQL = """
DELETE FROM captured_text_generated_views
WHERE event_id NOT IN (
    SELECT event_id FROM captured_text_events
);
"""

DELETE_ORPHANED_VIEW_GENERATION_JOBS_SQL = """
DELETE FROM captured_text_view_generation_jobs
WHERE event_id NOT IN (
    SELECT event_id FROM captured_text_events
);
"""

DELETE_ORPHANED_ANALYSIS_JOBS_SQL = """
DELETE FROM captured_text_analysis_jobs
WHERE event_id NOT IN (
    SELECT event_id FROM captured_text_events
);
"""


def delete_older_than(conn: sqlite3.Connection, *, cutoff: datetime) -> int:
    cursor = conn.execute(DELETE_OLDER_THAN_SQL, (cutoff.isoformat(),))
    delete_orphaned_rows(conn)
    return max(cursor.rowcount, 0)


def delete_oldest_excess(conn: sqlite3.Connection, *, keep_latest: int) -> int:
    cursor = conn.execute(DELETE_EXCESS_SQL, (keep_latest,))
    delete_orphaned_rows(conn)
    return max(cursor.rowcount, 0)


def delete_orphaned_rows(conn: sqlite3.Connection) -> None:
    conn.execute(DELETE_ORPHANED_GENERATED_VIEWS_SQL)
    conn.execute(DELETE_ORPHANED_VIEW_GENERATION_JOBS_SQL)
    conn.execute(DELETE_ORPHANED_ANALYSIS_JOBS_SQL)
