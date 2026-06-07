"""agent-local 학습 입력 사용 이력 저장소."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

TRAINING_USAGE_RUN_V1 = "training_usage_run.v1"
TRAINING_USAGE_ROW_V1 = "training_usage_row.v1"
TRAINING_USAGE_STATUS_UPLOADED = "uploaded"
TRAINING_USAGE_ROLE_LABELED_ANCHOR = "labeled_anchor"
TRAINING_USAGE_ROLE_UNLABELED_GENERATED_VIEW = "unlabeled_generated_view"
TRAINING_USAGE_STAGE_QUERY_SSL_INPUT = "query_ssl_input"

_DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "training_usage_ledger.db"

_CREATE_RUN_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS training_usage_runs (
    update_id                TEXT PRIMARY KEY,
    schema_version           TEXT NOT NULL,
    round_id                 TEXT NOT NULL,
    task_id                  TEXT NOT NULL,
    recorded_at              TEXT NOT NULL,
    agent_id                 TEXT,
    model_id                 TEXT NOT NULL,
    model_revision           TEXT NOT NULL,
    objective_method_name    TEXT,
    objective_algorithm_name TEXT,
    status                   TEXT NOT NULL,
    candidate_count          INTEGER NOT NULL,
    accepted_count           INTEGER NOT NULL,
    metadata                 TEXT NOT NULL
);
"""

_CREATE_ROW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS training_usage_rows (
    update_id      TEXT NOT NULL,
    source_id      TEXT NOT NULL,
    role           TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    round_id       TEXT NOT NULL,
    task_id        TEXT NOT NULL,
    recorded_at    TEXT NOT NULL,
    source_kind    TEXT NOT NULL,
    stage          TEXT NOT NULL,
    label          TEXT,
    metadata       TEXT NOT NULL,
    PRIMARY KEY (update_id, source_id, role),
    FOREIGN KEY (update_id)
        REFERENCES training_usage_runs (update_id)
        ON DELETE CASCADE
);
"""

_CREATE_RUN_ROUND_TASK_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_training_usage_runs_round_task
ON training_usage_runs (round_id, task_id);
"""

_CREATE_ROW_ROUND_TASK_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_training_usage_rows_round_task
ON training_usage_rows (round_id, task_id);
"""

_INSERT_RUN_SQL = """
INSERT OR REPLACE INTO training_usage_runs
    (update_id, schema_version, round_id, task_id, recorded_at, agent_id,
     model_id, model_revision, objective_method_name, objective_algorithm_name,
     status, candidate_count, accepted_count, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_INSERT_ROW_SQL = """
INSERT OR REPLACE INTO training_usage_rows
    (update_id, source_id, role, schema_version, round_id, task_id, recorded_at,
     source_kind, stage, label, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RUN_SQL = """
SELECT update_id, schema_version, round_id, task_id, recorded_at, agent_id,
       model_id, model_revision, objective_method_name, objective_algorithm_name,
       status, candidate_count, accepted_count, metadata
FROM training_usage_runs
WHERE update_id = ?;
"""

_SELECT_RUNS_FOR_TASK_SQL = """
SELECT update_id, schema_version, round_id, task_id, recorded_at, agent_id,
       model_id, model_revision, objective_method_name, objective_algorithm_name,
       status, candidate_count, accepted_count, metadata
FROM training_usage_runs
WHERE round_id = ? AND task_id = ?
ORDER BY recorded_at ASC;
"""

_SELECT_ROWS_FOR_UPDATE_SQL = """
SELECT update_id, source_id, role, schema_version, round_id, task_id, recorded_at,
       source_kind, stage, label, metadata
FROM training_usage_rows
WHERE update_id = ?
ORDER BY role ASC, source_id ASC;
"""

_COUNT_ROWS_SQL = "SELECT COUNT(*) FROM training_usage_rows;"


@dataclass(slots=True, frozen=True)
class TrainingUsageRunRecord:
    """한 update가 어떤 학습 입력 묶음에서 만들어졌는지 남기는 run record."""

    update_id: str
    round_id: str
    task_id: str
    recorded_at: datetime
    model_id: str
    model_revision: str
    status: str
    candidate_count: int
    accepted_count: int
    agent_id: str | None = None
    objective_method_name: str | None = None
    objective_algorithm_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRAINING_USAGE_RUN_V1

    def __post_init__(self) -> None:
        _require_non_empty(self.update_id, "update_id")
        _require_non_empty(self.round_id, "round_id")
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.model_revision, "model_revision")
        _require_non_empty(self.status, "status")
        _require_non_empty(self.schema_version, "schema_version")
        if self.candidate_count < 0:
            raise ValueError("candidate_count must not be negative.")
        if self.accepted_count < 0:
            raise ValueError("accepted_count must not be negative.")


@dataclass(slots=True, frozen=True)
class TrainingUsageRowRecord:
    """한 update가 참조한 agent-local row 사용 기록."""

    update_id: str
    source_id: str
    role: str
    round_id: str
    task_id: str
    recorded_at: datetime
    source_kind: str
    stage: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRAINING_USAGE_ROW_V1

    def __post_init__(self) -> None:
        _require_non_empty(self.update_id, "update_id")
        _require_non_empty(self.source_id, "source_id")
        _require_non_empty(self.role, "role")
        _require_non_empty(self.round_id, "round_id")
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.source_kind, "source_kind")
        _require_non_empty(self.stage, "stage")
        _require_non_empty(self.schema_version, "schema_version")


@dataclass(slots=True)
class TrainingUsageLedgerRepository:
    """학습 입력 사용 이력을 agent-local SQLite에 저장한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(_CREATE_RUN_TABLE_SQL)
            conn.execute(_CREATE_ROW_TABLE_SQL)
            conn.execute(_CREATE_RUN_ROUND_TASK_INDEX_SQL)
            conn.execute(_CREATE_ROW_ROUND_TASK_INDEX_SQL)

    def save_run(
        self,
        run: TrainingUsageRunRecord,
        *,
        rows: tuple[TrainingUsageRowRecord, ...],
    ) -> None:
        """run record와 그 입력 row 사용 기록을 함께 저장한다."""

        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                _INSERT_RUN_SQL,
                (
                    run.update_id,
                    run.schema_version,
                    run.round_id,
                    run.task_id,
                    run.recorded_at.isoformat(),
                    run.agent_id,
                    run.model_id,
                    run.model_revision,
                    run.objective_method_name,
                    run.objective_algorithm_name,
                    run.status,
                    run.candidate_count,
                    run.accepted_count,
                    json.dumps(run.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.executemany(
                _INSERT_ROW_SQL,
                (
                    (
                        row.update_id,
                        row.source_id,
                        row.role,
                        row.schema_version,
                        row.round_id,
                        row.task_id,
                        row.recorded_at.isoformat(),
                        row.source_kind,
                        row.stage,
                        row.label,
                        json.dumps(row.metadata, ensure_ascii=False, sort_keys=True),
                    )
                    for row in rows
                ),
            )

    def get_run(self, update_id: str) -> TrainingUsageRunRecord | None:
        """update_id로 단일 run record를 읽는다."""

        with self._connect() as conn:
            row = conn.execute(_SELECT_RUN_SQL, (update_id,)).fetchone()
        return None if row is None else _row_to_run_record(row)

    def get_runs_for_task(
        self,
        *,
        round_id: str,
        task_id: str,
    ) -> tuple[TrainingUsageRunRecord, ...]:
        """round/task에 연결된 usage run 목록을 읽는다."""

        with self._connect() as conn:
            rows = conn.execute(
                _SELECT_RUNS_FOR_TASK_SQL,
                (round_id, task_id),
            ).fetchall()
        return tuple(_row_to_run_record(row) for row in rows)

    def get_rows_for_update(
        self,
        update_id: str,
    ) -> tuple[TrainingUsageRowRecord, ...]:
        """한 update가 사용한 row 기록을 읽는다."""

        with self._connect() as conn:
            rows = conn.execute(_SELECT_ROWS_FOR_UPDATE_SQL, (update_id,)).fetchall()
        return tuple(_row_to_usage_row_record(row) for row in rows)

    def count_rows(self) -> int:
        """저장된 row 사용 기록 수를 반환한다."""

        with self._connect() as conn:
            return int(conn.execute(_COUNT_ROWS_SQL).fetchone()[0])

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


def _row_to_run_record(row: tuple[Any, ...]) -> TrainingUsageRunRecord:
    (
        update_id,
        schema_version,
        round_id,
        task_id,
        recorded_at,
        agent_id,
        model_id,
        model_revision,
        objective_method_name,
        objective_algorithm_name,
        status,
        candidate_count,
        accepted_count,
        metadata,
    ) = row
    return TrainingUsageRunRecord(
        update_id=str(update_id),
        schema_version=str(schema_version),
        round_id=str(round_id),
        task_id=str(task_id),
        recorded_at=datetime.fromisoformat(str(recorded_at)),
        agent_id=None if agent_id is None else str(agent_id),
        model_id=str(model_id),
        model_revision=str(model_revision),
        objective_method_name=(
            None if objective_method_name is None else str(objective_method_name)
        ),
        objective_algorithm_name=(
            None if objective_algorithm_name is None else str(objective_algorithm_name)
        ),
        status=str(status),
        candidate_count=int(candidate_count),
        accepted_count=int(accepted_count),
        metadata=json.loads(str(metadata)),
    )


def _row_to_usage_row_record(row: tuple[Any, ...]) -> TrainingUsageRowRecord:
    (
        update_id,
        source_id,
        role,
        schema_version,
        round_id,
        task_id,
        recorded_at,
        source_kind,
        stage,
        label,
        metadata,
    ) = row
    return TrainingUsageRowRecord(
        update_id=str(update_id),
        source_id=str(source_id),
        role=str(role),
        schema_version=str(schema_version),
        round_id=str(round_id),
        task_id=str(task_id),
        recorded_at=datetime.fromisoformat(str(recorded_at)),
        source_kind=str(source_kind),
        stage=str(stage),
        label=None if label is None else str(label),
        metadata=json.loads(str(metadata)),
    )


def _require_non_empty(value: str, field_name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} must not be empty.")
