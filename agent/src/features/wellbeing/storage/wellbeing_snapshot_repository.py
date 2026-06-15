"""Wellbeing signal summary snapshot SQLite 저장소."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.src.contracts.wellbeing_signal_contracts import (
    DEFAULT_PARENT_WELLBEING_GUIDANCE,
    ParentWellbeingGuidancePayload,
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)
from agent.src.features.wellbeing.storage.wellbeing_storage import (
    DEFAULT_WELLBEING_DB_PATH,
    connect_wellbeing_db,
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wellbeing_snapshots (
    computed_at     TEXT PRIMARY KEY,
    schema_version  TEXT NOT NULL,
    projection_version TEXT NOT NULL DEFAULT 'legacy',
    signal_score    REAL NOT NULL,
    signal_level    TEXT NOT NULL,
    signal_label    TEXT NOT NULL,
    trend           TEXT NOT NULL,
    summary         TEXT NOT NULL,
    action_tip      TEXT NOT NULL,
    parent_response_priority TEXT NOT NULL,
    parent_conversation_starter TEXT NOT NULL,
    parent_caution_note TEXT NOT NULL,
    confidence      TEXT NOT NULL,
    low_data        INTEGER NOT NULL
);
"""

_ADD_PROJECTION_VERSION_COLUMN_SQL = """
ALTER TABLE wellbeing_snapshots
ADD COLUMN projection_version TEXT NOT NULL DEFAULT 'legacy';
"""

_PARENT_GUIDANCE_COLUMN_DEFAULTS = {
    "parent_response_priority": DEFAULT_PARENT_WELLBEING_GUIDANCE.response_priority,
    "parent_conversation_starter": (
        DEFAULT_PARENT_WELLBEING_GUIDANCE.conversation_starter
    ),
    "parent_caution_note": DEFAULT_PARENT_WELLBEING_GUIDANCE.caution_note,
}

_INSERT_SQL = """
INSERT OR REPLACE INTO wellbeing_snapshots
    (computed_at, schema_version, projection_version, signal_score,
     signal_level, signal_label, trend, summary, action_tip,
     parent_response_priority, parent_conversation_starter, parent_caution_note,
     confidence, low_data)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_LATEST_SQL = """
SELECT computed_at, schema_version, signal_score, signal_level, signal_label,
       trend, summary, action_tip, parent_response_priority,
       parent_conversation_starter, parent_caution_note, confidence, low_data
FROM wellbeing_snapshots
ORDER BY computed_at DESC
LIMIT 1;
"""

_SELECT_LATEST_PROJECTION_VERSION_SQL = """
SELECT projection_version
FROM wellbeing_snapshots
ORDER BY computed_at DESC
LIMIT 1;
"""

_SELECT_SINCE_SQL = """
SELECT computed_at, schema_version, signal_score, signal_level, signal_label,
       trend, summary, action_tip, parent_response_priority,
       parent_conversation_starter, parent_caution_note, confidence, low_data
FROM wellbeing_snapshots
WHERE computed_at >= ?
ORDER BY computed_at ASC;
"""


@dataclass(slots=True)
class WellbeingSnapshotRepository:
    """WellbeingSignalSummaryPayload를 로컬 SQLite에 저장한다."""

    db_path: Path = DEFAULT_WELLBEING_DB_PATH

    def __post_init__(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            _ensure_projection_version_column(conn)
            _ensure_parent_guidance_columns(conn)

    def save_summary(
        self,
        payload: WellbeingSignalSummaryPayload,
        *,
        projection_version: str = "legacy",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    payload.computed_at.isoformat(),
                    payload.schema_version,
                    projection_version,
                    payload.signal_score,
                    payload.signal_level.value,
                    payload.signal_label,
                    payload.trend.value,
                    payload.summary,
                    payload.action_tip,
                    payload.parent_guidance.response_priority,
                    payload.parent_guidance.conversation_starter,
                    payload.parent_guidance.caution_note,
                    payload.confidence.value,
                    1 if payload.low_data else 0,
                ),
            )

    def load_latest_summary(self) -> WellbeingSignalSummaryPayload | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_LATEST_SQL).fetchone()
        return None if row is None else _row_to_summary_payload(row)

    def load_latest_projection_version(self) -> str | None:
        """가장 최근 snapshot을 만든 projection logic version을 반환한다."""

        with self._connect() as conn:
            row = conn.execute(_SELECT_LATEST_PROJECTION_VERSION_SQL).fetchone()
        return None if row is None else str(row[0])

    def list_summaries_since(
        self,
        *,
        cutoff: datetime,
    ) -> tuple[WellbeingSignalSummaryPayload, ...]:
        with self._connect() as conn:
            rows = conn.execute(_SELECT_SINCE_SQL, (cutoff.isoformat(),)).fetchall()
        return tuple(_row_to_summary_payload(row) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        return connect_wellbeing_db(self.db_path)


def _ensure_projection_version_column(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(wellbeing_snapshots)")}
    if "projection_version" in columns:
        return
    conn.execute(_ADD_PROJECTION_VERSION_COLUMN_SQL)


def _ensure_parent_guidance_columns(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(wellbeing_snapshots)")}
    for column_name, default_value in _PARENT_GUIDANCE_COLUMN_DEFAULTS.items():
        if column_name in columns:
            continue
        escaped_default = default_value.replace("'", "''")
        conn.execute(
            "ALTER TABLE wellbeing_snapshots "
            f"ADD COLUMN {column_name} TEXT NOT NULL DEFAULT '{escaped_default}';"
        )


def _row_to_summary_payload(row: tuple[object, ...]) -> WellbeingSignalSummaryPayload:
    (
        computed_at,
        schema_version,
        signal_score,
        signal_level,
        signal_label,
        trend,
        summary,
        action_tip,
        parent_response_priority,
        parent_conversation_starter,
        parent_caution_note,
        confidence,
        low_data,
    ) = row
    return WellbeingSignalSummaryPayload(
        schema_version=str(schema_version),
        computed_at=datetime.fromisoformat(str(computed_at)),
        signal_score=float(signal_score),
        signal_level=WellbeingSignalLevel(str(signal_level)),
        signal_label=str(signal_label),
        trend=WellbeingSignalTrend(str(trend)),
        summary=str(summary),
        action_tip=str(action_tip),
        parent_guidance=ParentWellbeingGuidancePayload(
            response_priority=str(parent_response_priority),
            conversation_starter=str(parent_conversation_starter),
            caution_note=str(parent_caution_note),
        ),
        confidence=WellbeingSignalConfidence(str(confidence)),
        low_data=bool(low_data),
    )
