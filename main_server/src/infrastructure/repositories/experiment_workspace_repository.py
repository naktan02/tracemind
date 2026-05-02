"""Developer experiment workspace SQLite metadata repository."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS_ROOT = MAIN_SERVER_ROOT / "state" / "experiments"

_CREATE_WORKSPACES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiment_workspaces (
    workspace_id       TEXT PRIMARY KEY,
    manifest_id        TEXT NOT NULL,
    track_name         TEXT NOT NULL,
    entrypoint_name    TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    manifest_path      TEXT NOT NULL,
    resolved_plan_path TEXT,
    latest_run_id      TEXT
);
"""

_UPSERT_WORKSPACE_SQL = """
INSERT OR REPLACE INTO experiment_workspaces
    (workspace_id, manifest_id, track_name, entrypoint_name, created_at, updated_at,
     manifest_path, resolved_plan_path, latest_run_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_WORKSPACE_SQL = """
SELECT workspace_id, manifest_id, track_name, entrypoint_name, created_at, updated_at,
       manifest_path, resolved_plan_path, latest_run_id
FROM experiment_workspaces
WHERE workspace_id = ?;
"""

_LIST_WORKSPACES_SQL = """
SELECT workspace_id, manifest_id, track_name, entrypoint_name, created_at, updated_at,
       manifest_path, resolved_plan_path, latest_run_id
FROM experiment_workspaces
ORDER BY updated_at DESC
LIMIT ?;
"""

_SET_LATEST_RUN_SQL = """
UPDATE experiment_workspaces
SET latest_run_id = ?, updated_at = ?
WHERE workspace_id = ?;
"""

_DELETE_WORKSPACE_SQL = """
DELETE FROM experiment_workspaces
WHERE workspace_id = ?;
"""


@dataclass(slots=True)
class StoredExperimentWorkspaceRecord:
    """SQLite에 저장되는 workspace metadata canonical record."""

    workspace_id: str
    manifest_id: str
    track_name: str
    entrypoint_name: str
    created_at: datetime
    updated_at: datetime
    manifest_path: Path
    resolved_plan_path: Path | None = None
    latest_run_id: str | None = None


@dataclass(slots=True)
class ExperimentWorkspaceRepository:
    """Experiment workspace metadata와 저장 경로를 관리한다."""

    experiments_root: Path = field(default_factory=lambda: DEFAULT_EXPERIMENTS_ROOT)

    def __post_init__(self) -> None:
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_WORKSPACES_TABLE_SQL)

    @property
    def db_path(self) -> Path:
        return self.experiments_root / "metadata.db"

    @property
    def workspaces_dir(self) -> Path:
        return self.experiments_root / "workspaces"

    def workspace_dir(self, workspace_id: str) -> Path:
        return self.workspaces_dir / workspace_id

    def save(self, record: StoredExperimentWorkspaceRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT_WORKSPACE_SQL,
                (
                    record.workspace_id,
                    record.manifest_id,
                    record.track_name,
                    record.entrypoint_name,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    str(record.manifest_path),
                    (
                        None
                        if record.resolved_plan_path is None
                        else str(record.resolved_plan_path)
                    ),
                    record.latest_run_id,
                ),
            )

    def get(self, workspace_id: str) -> StoredExperimentWorkspaceRecord | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_WORKSPACE_SQL, (workspace_id,)).fetchone()
        return None if row is None else _workspace_row_to_record(row)

    def list_recent(self, *, limit: int = 20) -> list[StoredExperimentWorkspaceRecord]:
        with self._connect() as conn:
            rows = conn.execute(_LIST_WORKSPACES_SQL, (limit,)).fetchall()
        return [_workspace_row_to_record(row) for row in rows]

    def set_latest_run(
        self,
        workspace_id: str,
        *,
        run_id: str,
        updated_at: datetime,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                _SET_LATEST_RUN_SQL,
                (
                    run_id,
                    updated_at.isoformat(),
                    workspace_id,
                ),
            )

    def delete(self, workspace_id: str) -> None:
        with self._connect() as conn:
            conn.execute(_DELETE_WORKSPACE_SQL, (workspace_id,))

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)


def _workspace_row_to_record(
    row: tuple[object, ...],
) -> StoredExperimentWorkspaceRecord:
    (
        workspace_id,
        manifest_id,
        track_name,
        entrypoint_name,
        created_at_str,
        updated_at_str,
        manifest_path,
        resolved_plan_path,
        latest_run_id,
    ) = row
    return StoredExperimentWorkspaceRecord(
        workspace_id=str(workspace_id),
        manifest_id=str(manifest_id),
        track_name=str(track_name),
        entrypoint_name=str(entrypoint_name),
        created_at=datetime.fromisoformat(str(created_at_str)),
        updated_at=datetime.fromisoformat(str(updated_at_str)),
        manifest_path=Path(str(manifest_path)),
        resolved_plan_path=(
            None if resolved_plan_path is None else Path(str(resolved_plan_path))
        ),
        latest_run_id=None if latest_run_id is None else str(latest_run_id),
    )
