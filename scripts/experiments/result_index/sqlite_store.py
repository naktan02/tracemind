"""SQLite writer for experiment result indexes."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.experiments.result_index.models import ResultIndexRecords
from scripts.experiments.result_index.schema import SCHEMA_STATEMENTS

_CHILD_TABLES = (
    "eval_metrics",
    "per_class_metrics",
    "confusion_matrix_cells",
    "epoch_metrics",
    "epoch_per_class_metrics",
)


def initialize_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)


def clear_database(db_path: Path) -> None:
    """Delete indexed rows while keeping the schema."""

    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        for table in _CHILD_TABLES:
            connection.execute(f"delete from {table}")
        connection.execute("delete from experiment_runs")


def write_result_index_records(
    *,
    db_path: Path,
    records: Sequence[ResultIndexRecords],
) -> None:
    """Upsert run records and replace their normalized child rows."""

    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        for result_records in records:
            _delete_run_children(connection, result_records.run.run_id)
            _insert_rows(
                connection,
                table="experiment_runs",
                rows=[asdict(result_records.run)],
                replace=True,
            )
            _insert_rows(
                connection,
                table="eval_metrics",
                rows=_as_dicts(result_records.eval_metrics),
            )
            _insert_rows(
                connection,
                table="per_class_metrics",
                rows=_as_dicts(result_records.per_class_metrics),
            )
            _insert_rows(
                connection,
                table="confusion_matrix_cells",
                rows=_as_dicts(result_records.confusion_matrix_cells),
            )
            _insert_rows(
                connection,
                table="epoch_metrics",
                rows=_as_dicts(result_records.epoch_metrics),
            )
            _insert_rows(
                connection,
                table="epoch_per_class_metrics",
                rows=_as_dicts(result_records.epoch_per_class_metrics),
            )


def _delete_run_children(connection: sqlite3.Connection, run_id: str) -> None:
    for table in _CHILD_TABLES:
        connection.execute(f"delete from {table} where run_id = ?", (run_id,))


def _insert_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    rows: Sequence[dict[str, Any]],
    replace: bool = False,
) -> None:
    if not rows:
        return
    columns = tuple(rows[0])
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)
    mode = "insert or replace" if replace else "insert"
    values = [tuple(row[column] for column in columns) for row in rows]
    connection.executemany(
        f"{mode} into {table} ({column_sql}) values ({placeholders})",
        values,
    )


def _as_dicts(records: Iterable[object]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
