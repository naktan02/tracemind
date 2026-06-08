"""QueryBuffer SQLite 저장소.

query-domain 적응 준비를 위해 raw query text와 예측 스냅샷을
agent 로컬에만 저장한다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.src.infrastructure.repositories.local_agent_database import (
    DEFAULT_AGENT_LOCAL_DB_PATH,
    connect_agent_local_db,
)
from shared.src.domain.entities.inference.events import AnalysisEvent, QueryEvent

QUERY_BUFFER_RECORD_V1 = "query_buffer_record.v1"

# 기본 DB 경로: agent 로컬 data 디렉토리
_DEFAULT_DB_PATH = DEFAULT_AGENT_LOCAL_DB_PATH

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_buffer_records (
    query_id         TEXT PRIMARY KEY,
    schema_version   TEXT NOT NULL,
    occurred_at      TEXT NOT NULL,
    raw_text         TEXT NOT NULL,
    locale           TEXT NOT NULL,
    source_type      TEXT NOT NULL,
    model_revision   TEXT NOT NULL,
    predicted_label  TEXT,
    confidence       REAL,
    margin           REAL,
    runner_up_label  TEXT,
    runner_up_score  REAL,
    confidence_kind  TEXT NOT NULL,
    metadata         TEXT NOT NULL
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO query_buffer_records
    (query_id, schema_version, occurred_at, raw_text, locale, source_type,
     model_revision, predicted_label, confidence, margin, runner_up_label,
     runner_up_score, confidence_kind, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_ONE_SQL = """
SELECT query_id, schema_version, occurred_at, raw_text, locale, source_type,
       model_revision, predicted_label, confidence, margin, runner_up_label,
       runner_up_score, confidence_kind, metadata
FROM query_buffer_records
WHERE query_id = ?;
"""

_SELECT_RECENT_SQL = """
SELECT query_id, schema_version, occurred_at, raw_text, locale, source_type,
       model_revision, predicted_label, confidence, margin, runner_up_label,
       runner_up_score, confidence_kind, metadata
FROM query_buffer_records
ORDER BY occurred_at DESC
LIMIT ?;
"""

_COUNT_SQL = "SELECT COUNT(*) FROM query_buffer_records;"

_DELETE_OLDER_THAN_SQL = """
DELETE FROM query_buffer_records
WHERE occurred_at < ?;
"""

_DELETE_EXCESS_SQL = """
DELETE FROM query_buffer_records
WHERE query_id IN (
    SELECT query_id
    FROM query_buffer_records
    ORDER BY occurred_at DESC
    LIMIT -1 OFFSET ?
);
"""


@dataclass(slots=True)
class QueryBufferRecord:
    """agent 로컬 query buffer의 canonical event snapshot."""

    query_id: str
    occurred_at: datetime
    raw_text: str
    locale: str
    source_type: str
    model_revision: str
    predicted_label: str | None
    confidence: float | None
    margin: float | None
    runner_up_label: str | None
    runner_up_score: float | None
    confidence_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QUERY_BUFFER_RECORD_V1

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be empty.")
        if not self.raw_text.strip():
            raise ValueError("raw_text must not be empty.")
        if not self.locale.strip():
            raise ValueError("locale must not be empty.")
        if not self.source_type.strip():
            raise ValueError("source_type must not be empty.")
        if not self.model_revision.strip():
            raise ValueError("model_revision must not be empty.")
        if not self.confidence_kind.strip():
            raise ValueError("confidence_kind must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty.")


def build_query_buffer_record(
    *,
    event: QueryEvent,
    analysis_event: AnalysisEvent,
    model_revision: str,
    confidence_kind: str,
    metadata: dict[str, Any] | None = None,
) -> QueryBufferRecord:
    """QueryEvent/AnalysisEvent 쌍에서 query buffer snapshot을 만든다."""

    ranked_scores = sorted(
        (
            (str(label), float(score))
            for label, score in analysis_event.category_scores.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    predicted_label = ranked_scores[0][0] if ranked_scores else None
    confidence = ranked_scores[0][1] if ranked_scores else None
    runner_up_label = ranked_scores[1][0] if len(ranked_scores) > 1 else None
    runner_up_score = ranked_scores[1][1] if len(ranked_scores) > 1 else None
    margin = (
        confidence - runner_up_score
        if confidence is not None and runner_up_score is not None
        else None
    )
    return QueryBufferRecord(
        query_id=event.query_id,
        occurred_at=event.occurred_at,
        raw_text=event.text,
        locale=event.locale,
        source_type=event.source_type,
        model_revision=model_revision,
        predicted_label=predicted_label,
        confidence=confidence,
        margin=margin,
        runner_up_label=runner_up_label,
        runner_up_score=runner_up_score,
        confidence_kind=confidence_kind,
        metadata={} if metadata is None else dict(metadata),
    )


@dataclass(slots=True)
class QueryBufferRepository:
    """QueryBufferRecord를 SQLite에 저장하고 최소 조회를 제공한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def save(self, record: QueryBufferRecord) -> None:
        """query_id 기준으로 query buffer snapshot을 저장한다."""

        with self._connect() as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    record.query_id,
                    record.schema_version,
                    record.occurred_at.isoformat(),
                    record.raw_text,
                    record.locale,
                    record.source_type,
                    record.model_revision,
                    record.predicted_label,
                    record.confidence,
                    record.margin,
                    record.runner_up_label,
                    record.runner_up_score,
                    record.confidence_kind,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )

    def get(self, query_id: str) -> QueryBufferRecord | None:
        """단일 query_id의 저장 레코드를 반환한다."""

        with self._connect() as conn:
            row = conn.execute(_SELECT_ONE_SQL, (query_id,)).fetchone()
        return None if row is None else _row_to_record(row)

    def get_recent(self, *, limit: int = 50) -> list[QueryBufferRecord]:
        """최근 query buffer 레코드를 최신순으로 반환한다."""

        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (limit,)).fetchall()
        return [_row_to_record(row) for row in rows]

    def count(self) -> int:
        """저장된 총 query buffer 레코드 수를 반환한다."""

        with self._connect() as conn:
            return conn.execute(_COUNT_SQL).fetchone()[0]

    def delete_older_than(self, *, cutoff: datetime) -> int:
        """cutoff보다 오래된 query buffer 레코드를 삭제한다."""

        with self._connect() as conn:
            cursor = conn.execute(_DELETE_OLDER_THAN_SQL, (cutoff.isoformat(),))
        return max(cursor.rowcount, 0)

    def delete_oldest_excess(self, *, keep_latest: int) -> int:
        """최신 keep_latest개를 제외한 오래된 레코드를 삭제한다."""

        if keep_latest < 0:
            raise ValueError("keep_latest must not be negative.")
        with self._connect() as conn:
            cursor = conn.execute(_DELETE_EXCESS_SQL, (keep_latest,))
        return max(cursor.rowcount, 0)

    def _connect(self) -> sqlite3.Connection:
        return connect_agent_local_db(self.db_path)


def _row_to_record(row: tuple[Any, ...]) -> QueryBufferRecord:
    (
        query_id,
        schema_version,
        occurred_at_str,
        raw_text,
        locale,
        source_type,
        model_revision,
        predicted_label,
        confidence,
        margin,
        runner_up_label,
        runner_up_score,
        confidence_kind,
        metadata_json,
    ) = row
    return QueryBufferRecord(
        query_id=str(query_id),
        schema_version=str(schema_version),
        occurred_at=datetime.fromisoformat(str(occurred_at_str)),
        raw_text=str(raw_text),
        locale=str(locale),
        source_type=str(source_type),
        model_revision=str(model_revision),
        predicted_label=None if predicted_label is None else str(predicted_label),
        confidence=None if confidence is None else float(confidence),
        margin=None if margin is None else float(margin),
        runner_up_label=None if runner_up_label is None else str(runner_up_label),
        runner_up_score=(None if runner_up_score is None else float(runner_up_score)),
        confidence_kind=str(confidence_kind),
        metadata=json.loads(str(metadata_json)),
    )
