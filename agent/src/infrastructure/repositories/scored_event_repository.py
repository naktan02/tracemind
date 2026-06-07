"""로컬 analysis event SQLite 저장소.

기존 서비스 호환을 위해 ScoredEventRepository 이름을 유지하지만,
실제 저장 schema는 scorer-family 독립적인 analysis_events 구조를 사용한다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shared.src.domain.entities.inference.events import ScoredEvent

# 기본 DB 경로: agent 로컬 data 디렉토리
_DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "analysis_events.db"

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
    confidence_kind      TEXT NOT NULL,
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
     confidence_kind, embedding_model_id, translation_model_id,
     base_embedding, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_INSERT_CATEGORY_SCORE_SQL = """
INSERT INTO analysis_category_scores
    (analysis_id, category, score)
VALUES (?, ?, ?);
"""

_SELECT_RECENT_SQL = """
SELECT analysis_id, occurred_at, translated_text,
       embedding_model_id, translation_model_id, base_embedding
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


@dataclass(slots=True)
class StoredScoredEvent:
    """SQLite에서 읽어온 ScoredEvent + base_embedding 묶음."""

    scored_event: ScoredEvent
    base_embedding: list[float] | None


@dataclass(slots=True)
class ScoredEventRepository:
    """analysis event 저장소의 기존 ScoredEvent 호환 facade."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_ANALYSIS_EVENTS_TABLE_SQL)
            conn.execute(_CREATE_ANALYSIS_CATEGORY_SCORES_TABLE_SQL)
            conn.execute(_CREATE_ANALYSIS_OCCURRED_AT_INDEX_SQL)
            conn.execute(_CREATE_ANALYSIS_SCORER_FAMILY_INDEX_SQL)
            conn.execute(_CREATE_ANALYSIS_CATEGORY_INDEX_SQL)

    # ------------------------------------------------------------------ #
    # 쓰기                                                                 #
    # ------------------------------------------------------------------ #

    def save(
        self,
        event: ScoredEvent,
        *,
        base_embedding: list[float] | None = None,
        source_event_id: str | None = None,
        scorer_family: str = "legacy_scored_event",
        scorer_name: str = "unknown",
        model_revision: str = "unknown",
        confidence_kind: str = "unknown",
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        """분석 결과를 저장한다. 기존 호출부는 ScoredEvent만 넘겨도 된다."""
        top_category, top_score = _get_top_category_score(event.category_scores)
        with self._connect() as conn:
            conn.execute(_DELETE_CATEGORY_SCORES_SQL, (event.query_id,))
            conn.execute(
                _INSERT_ANALYSIS_EVENT_SQL,
                (
                    event.query_id,
                    source_event_id or event.query_id,
                    event.occurred_at.isoformat(),
                    event.translated_text,
                    scorer_family,
                    scorer_name,
                    model_revision,
                    top_category,
                    top_score,
                    confidence_kind,
                    event.embedding_model_id,
                    event.translation_model_id,
                    json.dumps(base_embedding) if base_embedding is not None else None,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            conn.executemany(
                _INSERT_CATEGORY_SCORE_SQL,
                (
                    (event.query_id, category, float(score))
                    for category, score in event.category_scores.items()
                ),
            )

    # ------------------------------------------------------------------ #
    # 읽기                                                                 #
    # ------------------------------------------------------------------ #

    def get_recent(self, *, days: int = 7) -> list[ScoredEvent]:
        """최근 N일 이내의 ScoredEvent 목록을 반환한다."""
        return [s.scored_event for s in self.get_recent_stored(days=days)]

    def get_recent_stored(self, *, days: int = 7) -> list[StoredScoredEvent]:
        """최근 N일 이내의 ScoredEvent와 base_embedding을 함께 반환한다."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (cutoff,)).fetchall()
            return [_row_to_stored(conn, row) for row in rows]

    def count(self) -> int:
        """저장된 총 이벤트 수를 반환한다."""
        with self._connect() as conn:
            return conn.execute(_COUNT_SQL).fetchone()[0]

    # ------------------------------------------------------------------ #
    # 내부                                                                 #
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


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


def _row_to_stored(conn: sqlite3.Connection, row: tuple) -> StoredScoredEvent:
    (
        query_id,
        occurred_at_str,
        translated_text,
        embedding_model_id,
        translation_model_id,
        base_embedding_json,
    ) = row
    category_scores = {
        category: float(score)
        for category, score in conn.execute(_SELECT_CATEGORY_SCORES_SQL, (query_id,))
    }
    scored_event = ScoredEvent(
        query_id=query_id,
        occurred_at=datetime.fromisoformat(occurred_at_str),
        translated_text=translated_text,
        embedding_model_id=embedding_model_id or "unknown",
        translation_model_id=translation_model_id,
        category_scores=category_scores,
    )
    base_embedding = (
        json.loads(base_embedding_json) if base_embedding_json is not None else None
    )
    return StoredScoredEvent(scored_event=scored_event, base_embedding=base_embedding)
