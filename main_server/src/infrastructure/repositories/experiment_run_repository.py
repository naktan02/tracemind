"""Developer experiment run SQLite metadata repository."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from main_server.src.infrastructure.repositories import (
    experiment_workspace_repository,
)

_CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id              TEXT PRIMARY KEY,
    workspace_id        TEXT,
    manifest_id         TEXT NOT NULL,
    track_name          TEXT NOT NULL,
    entrypoint_name     TEXT NOT NULL,
    status              TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    script_path         TEXT NOT NULL,
    command_args_json   TEXT NOT NULL,
    manifest_path       TEXT NOT NULL,
    resolved_plan_path  TEXT NOT NULL,
    artifact_root_path  TEXT NOT NULL,
    stdout_log_path     TEXT NOT NULL,
    stderr_log_path     TEXT NOT NULL,
    pid                 INTEGER,
    exit_code           INTEGER,
    error_message       TEXT
);
"""

_UPSERT_RUN_SQL = """
INSERT OR REPLACE INTO experiment_runs
    (run_id, workspace_id, manifest_id, track_name, entrypoint_name, status,
     created_at, started_at, finished_at, script_path, command_args_json,
     manifest_path, resolved_plan_path, artifact_root_path, stdout_log_path,
     stderr_log_path, pid, exit_code, error_message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RUN_SQL = """
SELECT run_id, workspace_id, manifest_id, track_name, entrypoint_name, status,
       created_at, started_at, finished_at, script_path, command_args_json,
       manifest_path, resolved_plan_path, artifact_root_path, stdout_log_path,
       stderr_log_path, pid, exit_code, error_message
FROM experiment_runs
WHERE run_id = ?;
"""

_LIST_RUNS_SQL = """
SELECT run_id, workspace_id, manifest_id, track_name, entrypoint_name, status,
       created_at, started_at, finished_at, script_path, command_args_json,
       manifest_path, resolved_plan_path, artifact_root_path, stdout_log_path,
       stderr_log_path, pid, exit_code, error_message
FROM experiment_runs
ORDER BY created_at DESC
LIMIT ?;
"""

_MARK_RUNNING_AS_INTERRUPTED_SQL = """
UPDATE experiment_runs
SET status = 'interrupted',
    finished_at = ?,
    error_message = ?
WHERE status = 'running';
"""


@dataclass(slots=True)
class StoredExperimentRunRecord:
    """SQLite에 저장되는 local run metadata canonical record."""

    run_id: str
    manifest_id: str
    track_name: str
    entrypoint_name: str
    status: str
    created_at: datetime
    started_at: datetime
    script_path: str
    command_args: tuple[str, ...]
    manifest_path: Path
    resolved_plan_path: Path
    artifact_root_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    workspace_id: str | None = None
    finished_at: datetime | None = None
    pid: int | None = None
    exit_code: int | None = None
    error_message: str | None = None


@dataclass(slots=True)
class ExperimentRunRepository:
    """Experiment run metadata와 저장 경로를 관리한다."""

    experiments_root: Path = field(
        default_factory=lambda: experiment_workspace_repository.DEFAULT_EXPERIMENTS_ROOT
    )

    def __post_init__(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS_TABLE_SQL)

    @property
    def db_path(self) -> Path:
        return self.experiments_root / "metadata.db"

    @property
    def runs_dir(self) -> Path:
        return self.experiments_root / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def save(self, record: StoredExperimentRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT_RUN_SQL,
                (
                    record.run_id,
                    record.workspace_id,
                    record.manifest_id,
                    record.track_name,
                    record.entrypoint_name,
                    record.status,
                    record.created_at.isoformat(),
                    record.started_at.isoformat(),
                    (
                        None
                        if record.finished_at is None
                        else record.finished_at.isoformat()
                    ),
                    record.script_path,
                    json.dumps(record.command_args),
                    str(record.manifest_path),
                    str(record.resolved_plan_path),
                    str(record.artifact_root_path),
                    str(record.stdout_log_path),
                    str(record.stderr_log_path),
                    record.pid,
                    record.exit_code,
                    record.error_message,
                ),
            )

    def get(self, run_id: str) -> StoredExperimentRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_RUN_SQL, (run_id,)).fetchone()
        return None if row is None else _run_row_to_record(row)

    def list_recent(self, *, limit: int = 20) -> list[StoredExperimentRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(_LIST_RUNS_SQL, (limit,)).fetchall()
        return [_run_row_to_record(row) for row in rows]

    def mark_running_as_interrupted(
        self,
        *,
        interrupted_at: datetime,
        error_message: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                _MARK_RUNNING_AS_INTERRUPTED_SQL,
                (interrupted_at.isoformat(), error_message),
            )
        return max(cursor.rowcount, 0)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)


def _run_row_to_record(row: tuple[object, ...]) -> StoredExperimentRunRecord:
    (
        run_id,
        workspace_id,
        manifest_id,
        track_name,
        entrypoint_name,
        status,
        created_at_str,
        started_at_str,
        finished_at_str,
        script_path,
        command_args_json,
        manifest_path,
        resolved_plan_path,
        artifact_root_path,
        stdout_log_path,
        stderr_log_path,
        pid,
        exit_code,
        error_message,
    ) = row
    return StoredExperimentRunRecord(
        run_id=str(run_id),
        workspace_id=None if workspace_id is None else str(workspace_id),
        manifest_id=str(manifest_id),
        track_name=str(track_name),
        entrypoint_name=str(entrypoint_name),
        status=str(status),
        created_at=datetime.fromisoformat(str(created_at_str)),
        started_at=datetime.fromisoformat(str(started_at_str)),
        finished_at=(
            None
            if finished_at_str is None
            else datetime.fromisoformat(str(finished_at_str))
        ),
        script_path=str(script_path),
        command_args=tuple(json.loads(str(command_args_json))),
        manifest_path=Path(str(manifest_path)),
        resolved_plan_path=Path(str(resolved_plan_path)),
        artifact_root_path=Path(str(artifact_root_path)),
        stdout_log_path=Path(str(stdout_log_path)),
        stderr_log_path=Path(str(stderr_log_path)),
        pid=None if pid is None else int(pid),
        exit_code=None if exit_code is None else int(exit_code),
        error_message=None if error_message is None else str(error_message),
    )
