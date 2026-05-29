"""Wellbeing extension 관련 최소 설정 SQLite 저장소."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent.src.infrastructure.repositories.wellbeing_storage import (
    DEFAULT_WELLBEING_DB_PATH,
    connect_wellbeing_db,
)
from shared.src.contracts.wellbeing_signal_contracts import WellbeingSignalRange

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wellbeing_settings (
    record_id                   INTEGER PRIMARY KEY CHECK (record_id = 1),
    default_timeseries_range    TEXT NOT NULL,
    child_session_minutes       INTEGER NOT NULL,
    child_lock_minutes          INTEGER NOT NULL,
    child_max_attempts          INTEGER NOT NULL,
    parent_session_minutes      INTEGER NOT NULL,
    parent_lock_minutes         INTEGER NOT NULL,
    parent_max_attempts         INTEGER NOT NULL
);
"""

_UPSERT_SQL = """
INSERT INTO wellbeing_settings
    (record_id, default_timeseries_range, child_session_minutes, child_lock_minutes,
     child_max_attempts, parent_session_minutes, parent_lock_minutes,
     parent_max_attempts)
VALUES (1, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(record_id) DO UPDATE SET
    default_timeseries_range = excluded.default_timeseries_range,
    child_session_minutes = excluded.child_session_minutes,
    child_lock_minutes = excluded.child_lock_minutes,
    child_max_attempts = excluded.child_max_attempts,
    parent_session_minutes = excluded.parent_session_minutes,
    parent_lock_minutes = excluded.parent_lock_minutes,
    parent_max_attempts = excluded.parent_max_attempts;
"""

_SELECT_SQL = """
SELECT default_timeseries_range, child_session_minutes, child_lock_minutes,
       child_max_attempts, parent_session_minutes, parent_lock_minutes,
       parent_max_attempts
FROM wellbeing_settings
WHERE record_id = 1;
"""

_ADD_CHILD_SESSION_COLUMN_SQL = """
ALTER TABLE wellbeing_settings
ADD COLUMN child_session_minutes INTEGER NOT NULL DEFAULT 10;
"""

_ADD_CHILD_LOCK_COLUMN_SQL = """
ALTER TABLE wellbeing_settings ADD COLUMN child_lock_minutes INTEGER NOT NULL DEFAULT 3;
"""

_ADD_CHILD_MAX_ATTEMPTS_COLUMN_SQL = """
ALTER TABLE wellbeing_settings ADD COLUMN child_max_attempts INTEGER NOT NULL DEFAULT 5;
"""


@dataclass(slots=True)
class WellbeingSettingsRecord:
    """가족용 확장 MVP의 최소 설정 record."""

    default_timeseries_range: WellbeingSignalRange = WellbeingSignalRange.LAST_7_DAYS
    child_session_minutes: int = 10
    child_lock_minutes: int = 3
    child_max_attempts: int = 5
    parent_session_minutes: int = 15
    parent_lock_minutes: int = 10
    parent_max_attempts: int = 5


@dataclass(slots=True)
class WellbeingSettingsRepository:
    """가족용 확장 관련 최소 설정을 로컬 SQLite에 저장한다."""

    db_path: Path = DEFAULT_WELLBEING_DB_PATH

    def __post_init__(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            _migrate_settings_columns(conn)

    def load_settings(self) -> WellbeingSettingsRecord | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_SQL).fetchone()
        return None if row is None else _row_to_settings_record(row)

    def save_settings(self, settings: WellbeingSettingsRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT_SQL,
                (
                    settings.default_timeseries_range.value,
                    settings.child_session_minutes,
                    settings.child_lock_minutes,
                    settings.child_max_attempts,
                    settings.parent_session_minutes,
                    settings.parent_lock_minutes,
                    settings.parent_max_attempts,
                ),
            )

    def load_or_default(self) -> WellbeingSettingsRecord:
        settings = self.load_settings()
        if settings is None:
            settings = WellbeingSettingsRecord()
            self.save_settings(settings)
        return settings

    def _connect(self) -> sqlite3.Connection:
        return connect_wellbeing_db(self.db_path)


def _row_to_settings_record(row: tuple[object, ...]) -> WellbeingSettingsRecord:
    (
        default_timeseries_range,
        child_session_minutes,
        child_lock_minutes,
        child_max_attempts,
        parent_session_minutes,
        parent_lock_minutes,
        parent_max_attempts,
    ) = row
    return WellbeingSettingsRecord(
        default_timeseries_range=WellbeingSignalRange(str(default_timeseries_range)),
        child_session_minutes=int(child_session_minutes),
        child_lock_minutes=int(child_lock_minutes),
        child_max_attempts=int(child_max_attempts),
        parent_session_minutes=int(parent_session_minutes),
        parent_lock_minutes=int(parent_lock_minutes),
        parent_max_attempts=int(parent_max_attempts),
    )


def _migrate_settings_columns(conn: sqlite3.Connection) -> None:
    table_info_rows = conn.execute("PRAGMA table_info(wellbeing_settings)").fetchall()
    columns = {row[1] for row in table_info_rows}
    if "child_session_minutes" not in columns:
        conn.execute(_ADD_CHILD_SESSION_COLUMN_SQL)
    if "child_lock_minutes" not in columns:
        conn.execute(_ADD_CHILD_LOCK_COLUMN_SQL)
    if "child_max_attempts" not in columns:
        conn.execute(_ADD_CHILD_MAX_ATTEMPTS_COLUMN_SQL)
