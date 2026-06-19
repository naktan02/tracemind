"""로컬 analysis event SQLite 저장소."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.local_agent_database import (
    DEFAULT_AGENT_LOCAL_DB_PATH,
    connect_agent_local_db,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

# 기본 DB 경로: agent 로컬 data 디렉토리
_DEFAULT_DB_PATH = DEFAULT_AGENT_LOCAL_DB_PATH

_CREATE_ANALYSIS_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analysis_events (
    analysis_id          TEXT PRIMARY KEY,
    source_event_id      TEXT NOT NULL,
    occurred_at          TEXT NOT NULL,
    translated_text      TEXT,
    scorer_family        TEXT NOT NULL,
    scorer_name          TEXT NOT NULL,
    model_revision       TEXT NOT NULL,
    top_category         TEXT,
    top_score            REAL,
    embedding_model_id   TEXT,
    translation_model_id TEXT,
    base_embedding       TEXT,
    metadata             TEXT NOT NULL
);
"""

_CREATE_ANALYSIS_CATEGORY_SCORES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analysis_category_scores (
    analysis_id TEXT NOT NULL,
    category    TEXT NOT NULL,
    score       REAL NOT NULL,
    PRIMARY KEY (analysis_id, category),
    FOREIGN KEY (analysis_id)
        REFERENCES analysis_events (analysis_id)
        ON DELETE CASCADE
);
"""

_CREATE_ANALYSIS_OCCURRED_AT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_analysis_events_occurred_at
ON analysis_events (occurred_at);
"""

_CREATE_ANALYSIS_SCORER_FAMILY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_analysis_events_scorer_family
ON analysis_events (scorer_family);
"""

_CREATE_ANALYSIS_CATEGORY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_analysis_category_scores_category
ON analysis_category_scores (category);
"""

_DELETE_CATEGORY_SCORES_SQL = """
DELETE FROM analysis_category_scores
WHERE analysis_id = ?;
"""

_INSERT_ANALYSIS_EVENT_SQL = """
INSERT OR REPLACE INTO analysis_events
    (analysis_id, source_event_id, occurred_at, translated_text,
     scorer_family, scorer_name, model_revision, top_category, top_score,
     embedding_model_id, translation_model_id, base_embedding, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_INSERT_CATEGORY_SCORE_SQL = """
INSERT INTO analysis_category_scores
    (analysis_id, category, score)
VALUES (?, ?, ?);
"""

_SELECT_RECENT_SQL = """
SELECT analysis_id, source_event_id, occurred_at, translated_text,
       scorer_family, scorer_name, model_revision, embedding_model_id,
       translation_model_id, base_embedding, metadata
FROM analysis_events
WHERE occurred_at >= ?
ORDER BY occurred_at DESC;
"""

_SELECT_CATEGORY_SCORES_SQL = """
SELECT category, score
FROM analysis_category_scores
WHERE analysis_id = ?
ORDER BY category ASC;
"""

_COUNT_SQL = "SELECT COUNT(*) FROM analysis_events;"

_SELECT_LATEST_OCCURRED_AT_SQL = """
SELECT occurred_at
FROM analysis_events
ORDER BY occurred_at DESC
LIMIT 1;
"""

_EXISTS_SOURCE_EVENT_SQL = """
SELECT 1
FROM analysis_events
WHERE source_event_id = ?
LIMIT 1;
"""


@dataclass(slots=True)
class StoredAnalysisEvent:
    """SQLite에서 읽어온 AnalysisEvent + base_embedding 묶음."""

    analysis_event: AnalysisEvent
    base_embedding: list[float] | None


@dataclass(slots=True)
class AnalysisEventRepository:
    """AnalysisEvent와 base_embedding을 SQLite에 저장하고 조회한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            ensure_analysis_event_schema(conn)

    # ------------------------------------------------------------------ #
    # 쓰기                                                                 #
    # ------------------------------------------------------------------ #

    def save(
        self,
        event: AnalysisEvent,
        *,
        base_embedding: list[float] | None = None,
        source_event_id: str | None = None,
        scorer_family: str = "unknown",
        scorer_name: str = "unknown",
        model_revision: str = "unknown",
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        """AnalysisEvent를 저장한다. base_embedding이 있으면 함께 저장한다."""
        top_category, top_score = _get_top_category_score(event.category_scores)
        analysis_id = event.analysis_id or event.query_id
        effective_source_event_id = (
            source_event_id or event.source_event_id or event.query_id
        )
        effective_scorer_family = (
            scorer_family if scorer_family != "unknown" else event.scorer_family
        )
        effective_scorer_name = (
            scorer_name if scorer_name != "unknown" else event.scorer_name
        )
        effective_model_revision = (
            model_revision if model_revision != "unknown" else event.model_revision
        )
        effective_metadata = {**event.metadata, **(metadata or {})}
        with self._connect() as conn:
            conn.execute(_DELETE_CATEGORY_SCORES_SQL, (analysis_id,))
            conn.execute(
                _INSERT_ANALYSIS_EVENT_SQL,
                (
                    analysis_id,
                    effective_source_event_id,
                    event.occurred_at.isoformat(),
                    event.translated_text,
                    effective_scorer_family,
                    effective_scorer_name,
                    effective_model_revision,
                    top_category,
                    top_score,
                    event.embedding_model_id,
                    event.translation_model_id,
                    json.dumps(base_embedding) if base_embedding is not None else None,
                    json.dumps(effective_metadata, sort_keys=True),
                ),
            )
            conn.executemany(
                _INSERT_CATEGORY_SCORE_SQL,
                (
                    (analysis_id, category, float(score))
                    for category, score in event.category_scores.items()
                ),
            )

    # ------------------------------------------------------------------ #
    # 읽기                                                                 #
    # ------------------------------------------------------------------ #

    def get_recent(self, *, days: int = 7) -> list[AnalysisEvent]:
        """최근 N일 이내의 AnalysisEvent 목록을 반환한다."""
        return [s.analysis_event for s in self.get_recent_stored(days=days)]

    def get_recent_stored(self, *, days: int = 7) -> list[StoredAnalysisEvent]:
        """최근 N일 이내의 AnalysisEvent와 base_embedding을 함께 반환한다."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (cutoff,)).fetchall()
            return [_row_to_stored(conn, row) for row in rows]

    def count(self) -> int:
        """저장된 총 이벤트 수를 반환한다."""
        with self._connect() as conn:
            return conn.execute(_COUNT_SQL).fetchone()[0]

    def load_latest_occurred_at(self) -> datetime | None:
        """가장 최근 analysis event 시각을 반환한다."""
        with self._connect() as conn:
            row = conn.execute(_SELECT_LATEST_OCCURRED_AT_SQL).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(str(row[0]))

    def has_source_event_id(self, source_event_id: str) -> bool:
        """source_event_id로 저장된 분석 결과가 이미 있는지 확인한다."""

        with self._connect() as conn:
            row = conn.execute(_EXISTS_SOURCE_EVENT_SQL, (source_event_id,)).fetchone()
        return row is not None

    # ------------------------------------------------------------------ #
    # 내부                                                                 #
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        return connect_agent_local_db(self.db_path)


def ensure_analysis_event_schema(conn: sqlite3.Connection) -> None:
    """analysis event 관련 table/index를 생성한다."""

    conn.execute(_CREATE_ANALYSIS_EVENTS_TABLE_SQL)
    _drop_legacy_confidence_kind_column(conn)
    conn.execute(_CREATE_ANALYSIS_CATEGORY_SCORES_TABLE_SQL)
    _repair_category_scores_legacy_fk(conn)
    conn.execute(_CREATE_ANALYSIS_OCCURRED_AT_INDEX_SQL)
    conn.execute(_CREATE_ANALYSIS_SCORER_FAMILY_INDEX_SQL)
    conn.execute(_CREATE_ANALYSIS_CATEGORY_INDEX_SQL)


# ------------------------------------------------------------------ #
# 내부 변환 함수                                                         #
# ------------------------------------------------------------------ #


def _get_top_category_score(
    category_scores: dict[str, float],
) -> tuple[str | None, float | None]:
    if not category_scores:
        return None, None
    top_category = max(category_scores, key=category_scores.__getitem__)
    return top_category, float(category_scores[top_category])


def _row_to_stored(conn: sqlite3.Connection, row: tuple) -> StoredAnalysisEvent:
    (
        analysis_id,
        source_event_id,
        occurred_at_str,
        translated_text,
        scorer_family,
        scorer_name,
        model_revision,
        embedding_model_id,
        translation_model_id,
        base_embedding_json,
        metadata_json,
    ) = row
    category_scores = {
        category: float(score)
        for category, score in conn.execute(_SELECT_CATEGORY_SCORES_SQL, (analysis_id,))
    }
    analysis_event = AnalysisEvent(
        query_id=source_event_id,
        occurred_at=datetime.fromisoformat(occurred_at_str),
        translated_text=translated_text,
        embedding_model_id=embedding_model_id or "unknown",
        translation_model_id=translation_model_id,
        category_scores=category_scores,
        analysis_id=analysis_id,
        source_event_id=source_event_id,
        scorer_family=scorer_family,
        scorer_name=scorer_name,
        model_revision=model_revision,
        metadata=json.loads(metadata_json),
    )
    base_embedding = (
        json.loads(base_embedding_json) if base_embedding_json is not None else None
    )
    return StoredAnalysisEvent(
        analysis_event=analysis_event,
        base_embedding=base_embedding,
    )


def _drop_legacy_confidence_kind_column(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(analysis_events);").fetchall()
    }
    if "confidence_kind" not in columns:
        return
    try:
        conn.execute("ALTER TABLE analysis_events DROP COLUMN confidence_kind;")
        return
    except sqlite3.OperationalError:
        pass

    category_rows = []
    category_table_exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'analysis_category_scores'
        LIMIT 1;
        """
    ).fetchone()
    if category_table_exists is not None:
        category_rows = conn.execute(
            """
            SELECT analysis_id, category, score
            FROM analysis_category_scores;
            """
        ).fetchall()
        conn.execute("DROP TABLE analysis_category_scores;")
    conn.execute("ALTER TABLE analysis_events RENAME TO analysis_events_legacy;")
    conn.execute(_CREATE_ANALYSIS_EVENTS_TABLE_SQL)
    conn.execute(
        """
        INSERT INTO analysis_events
            (analysis_id, source_event_id, occurred_at, translated_text,
             scorer_family, scorer_name, model_revision, top_category, top_score,
             embedding_model_id, translation_model_id, base_embedding, metadata)
        SELECT analysis_id, source_event_id, occurred_at, translated_text,
               scorer_family, scorer_name, model_revision, top_category, top_score,
               embedding_model_id, translation_model_id, base_embedding, metadata
        FROM analysis_events_legacy;
        """
    )
    conn.execute("DROP TABLE analysis_events_legacy;")
    if category_table_exists is not None:
        conn.execute(_CREATE_ANALYSIS_CATEGORY_SCORES_TABLE_SQL)
        conn.executemany(_INSERT_CATEGORY_SCORE_SQL, category_rows)


def _repair_category_scores_legacy_fk(conn: sqlite3.Connection) -> None:
    table_exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'analysis_category_scores'
        LIMIT 1;
        """
    ).fetchone()
    if table_exists is None:
        return

    fk_parent_tables = {
        str(row[2])
        for row in conn.execute(
            "PRAGMA foreign_key_list(analysis_category_scores)"
        ).fetchall()
    }
    if "analysis_events_legacy" not in fk_parent_tables:
        return

    rows = conn.execute(
        """
        SELECT s.analysis_id, s.category, s.score
        FROM analysis_category_scores s
        JOIN analysis_events e ON e.analysis_id = s.analysis_id;
        """
    ).fetchall()
    conn.execute("DROP TABLE analysis_category_scores;")
    conn.execute(_CREATE_ANALYSIS_CATEGORY_SCORES_TABLE_SQL)
    conn.executemany(_INSERT_CATEGORY_SCORE_SQL, rows)
