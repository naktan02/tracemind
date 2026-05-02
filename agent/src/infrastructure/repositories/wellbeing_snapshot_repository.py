"""Wellbeing signal summary snapshot SQLite 저장소."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.src.infrastructure.repositories.wellbeing_storage import (
    DEFAULT_WELLBEING_DB_PATH,
    connect_wellbeing_db,
)
from shared.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wellbeing_snapshots (
    computed_at     TEXT PRIMARY KEY,
    schema_version  TEXT NOT NULL,
    signal_score    REAL NOT NULL,
    signal_level    TEXT NOT NULL,
    signal_label    TEXT NOT NULL,
    trend           TEXT NOT NULL,
    summary         TEXT NOT NULL,
    action_tip      TEXT NOT NULL,
    confidence      TEXT NOT NULL,
    low_data        INTEGER NOT NULL
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO wellbeing_snapshots
    (computed_at, schema_version, signal_score, signal_level, signal_label,
     trend, summary, action_tip, confidence, low_data)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_LATEST_SQL = """
SELECT computed_at, schema_version, signal_score, signal_level, signal_label,
       trend, summary, action_tip, confidence, low_data
FROM wellbeing_snapshots
ORDER BY computed_at DESC
LIMIT 1;
"""

_SELECT_SINCE_SQL = """
SELECT computed_at, schema_version, signal_score, signal_level, signal_label,
       trend, summary, action_tip, confidence, low_data
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

    def save_summary(self, payload: WellbeingSignalSummaryPayload) -> None:
        with self._connect() as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    payload.computed_at.isoformat(),
                    payload.schema_version,
                    payload.signal_score,
                    payload.signal_level.value,
                    payload.signal_label,
                    payload.trend.value,
                    payload.summary,
                    payload.action_tip,
                    payload.confidence.value,
                    1 if payload.low_data else 0,
                ),
            )

    def load_latest_summary(self) -> WellbeingSignalSummaryPayload | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_LATEST_SQL).fetchone()
        return None if row is None else _row_to_summary_payload(row)

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
        confidence=WellbeingSignalConfidence(str(confidence)),
        low_data=bool(low_data),
    )
