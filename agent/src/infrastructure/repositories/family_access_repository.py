"""child/parent role PIN 상태 SQLite 저장소."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.src.contracts.family_access_contracts import FamilyAccessRole
from agent.src.infrastructure.repositories.wellbeing_storage import (
    DEFAULT_WELLBEING_DB_PATH,
    connect_wellbeing_db,
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS family_access_profiles (
    role                   TEXT PRIMARY KEY,
    pin_hash               TEXT NOT NULL,
    failed_attempt_count   INTEGER NOT NULL,
    locked_until           TEXT,
    updated_at             TEXT NOT NULL
);
"""

_UPSERT_SQL = """
INSERT INTO family_access_profiles
    (role, pin_hash, failed_attempt_count, locked_until, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(role) DO UPDATE SET
    pin_hash = excluded.pin_hash,
    failed_attempt_count = excluded.failed_attempt_count,
    locked_until = excluded.locked_until,
    updated_at = excluded.updated_at;
"""

_SELECT_BY_ROLE_SQL = """
SELECT role, pin_hash, failed_attempt_count, locked_until, updated_at
FROM family_access_profiles
WHERE role = ?;
"""

_SELECT_ALL_ROLES_SQL = """
SELECT role
FROM family_access_profiles
ORDER BY role ASC;
"""


@dataclass(slots=True)
class FamilyAccessState:
    """role별 PIN 잠금 상태 canonical record."""

    role: FamilyAccessRole
    pin_hash: str
    failed_attempt_count: int
    locked_until: datetime | None
    updated_at: datetime


@dataclass(slots=True)
class FamilyAccessRepository:
    """child/parent role PIN 상태를 로컬 SQLite에 저장한다."""

    db_path: Path = DEFAULT_WELLBEING_DB_PATH

    def __post_init__(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def list_configured_roles(self) -> tuple[FamilyAccessRole, ...]:
        with self._connect() as conn:
            rows = conn.execute(_SELECT_ALL_ROLES_SQL).fetchall()
        return tuple(FamilyAccessRole(str(row[0])) for row in rows)

    def load_state(self, role: FamilyAccessRole) -> FamilyAccessState | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_BY_ROLE_SQL, (role.value,)).fetchone()
        return None if row is None else _row_to_family_access_state(row)

    def save_state(self, state: FamilyAccessState) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT_SQL,
                (
                    state.role.value,
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


def _row_to_family_access_state(row: tuple[object, ...]) -> FamilyAccessState:
    role, pin_hash, failed_attempt_count, locked_until, updated_at = row
    return FamilyAccessState(
        role=FamilyAccessRole(str(role)),
        pin_hash=str(pin_hash),
        failed_attempt_count=int(failed_attempt_count),
        locked_until=(
            None if locked_until is None else datetime.fromisoformat(str(locked_until))
        ),
        updated_at=datetime.fromisoformat(str(updated_at)),
    )
