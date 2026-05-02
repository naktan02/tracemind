"""부모용 PIN 상태 SQLite 저장소."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.src.infrastructure.repositories.wellbeing_storage import (
    DEFAULT_WELLBEING_DB_PATH,
    connect_wellbeing_db,
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS parent_auth (
    record_id              INTEGER PRIMARY KEY CHECK (record_id = 1),
    pin_hash               TEXT NOT NULL,
    failed_attempt_count   INTEGER NOT NULL,
    locked_until           TEXT,
    updated_at             TEXT NOT NULL
);
"""

_UPSERT_SQL = """
INSERT INTO parent_auth
    (record_id, pin_hash, failed_attempt_count, locked_until, updated_at)
VALUES (1, ?, ?, ?, ?)
ON CONFLICT(record_id) DO UPDATE SET
    pin_hash = excluded.pin_hash,
    failed_attempt_count = excluded.failed_attempt_count,
    locked_until = excluded.locked_until,
    updated_at = excluded.updated_at;
"""

_SELECT_SQL = """
SELECT pin_hash, failed_attempt_count, locked_until, updated_at
FROM parent_auth
WHERE record_id = 1;
"""


@dataclass(slots=True)
class ParentAuthState:
    """부모용 PIN 잠금 상태 canonical record."""

    pin_hash: str
    failed_attempt_count: int
    locked_until: datetime | None
    updated_at: datetime


@dataclass(slots=True)
class ParentAuthRepository:
    """부모용 PIN 상태를 로컬 SQLite에 저장한다."""

    db_path: Path = DEFAULT_WELLBEING_DB_PATH

    def __post_init__(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def load_state(self) -> ParentAuthState | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_SQL).fetchone()
        return None if row is None else _row_to_parent_auth_state(row)

    def save_state(self, state: ParentAuthState) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT_SQL,
                (
                    state.pin_hash,
                    state.failed_attempt_count,
                    (
                        None
                        if state.locked_until is None
                        else state.locked_until.isoformat()
                    ),
                    state.updated_at.isoformat(),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        return connect_wellbeing_db(self.db_path)


def _row_to_parent_auth_state(row: tuple[object, ...]) -> ParentAuthState:
    pin_hash, failed_attempt_count, locked_until, updated_at = row
    return ParentAuthState(
        pin_hash=str(pin_hash),
        failed_attempt_count=int(failed_attempt_count),
        locked_until=(
            None if locked_until is None else datetime.fromisoformat(str(locked_until))
        ),
        updated_at=datetime.fromisoformat(str(updated_at)),
    )
