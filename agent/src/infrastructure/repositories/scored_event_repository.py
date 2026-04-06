"""ScoredEvent SQLite 저장소.

scored event를 로컬에 영속 저장하고 학습 시 조회한다.
base_embedding을 함께 저장해서 학습 시 재임베딩 없이 EmbeddedTrainingExample을 조립 가능하다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shared.src.domain.entities.inference.events import ScoredEvent

# 기본 DB 경로: agent 로컬 data 디렉토리
_DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "scored_events.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scored_events (
    query_id             TEXT PRIMARY KEY,
    occurred_at          TEXT NOT NULL,
    translated_text      TEXT,
    embedding_model_id   TEXT NOT NULL,
    translation_model_id TEXT,
    category_scores      TEXT NOT NULL,
    base_embedding       TEXT
);
"""

# 기존 DB 마이그레이션: base_embedding 컬럼이 없으면 추가
_ADD_EMBEDDING_COLUMN_SQL = """
ALTER TABLE scored_events ADD COLUMN base_embedding TEXT;
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO scored_events
    (query_id, occurred_at, translated_text,
     embedding_model_id, translation_model_id,
     category_scores, base_embedding)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT_SQL = """
SELECT query_id, occurred_at, translated_text,
       embedding_model_id, translation_model_id,
       category_scores, base_embedding
FROM scored_events
WHERE occurred_at >= ?
ORDER BY occurred_at DESC;
"""

_COUNT_SQL = "SELECT COUNT(*) FROM scored_events;"


@dataclass(slots=True)
class StoredScoredEvent:
    """SQLite에서 읽어온 ScoredEvent + base_embedding 묶음."""

    scored_event: ScoredEvent
    base_embedding: list[float] | None


@dataclass(slots=True)
class ScoredEventRepository:
    """ScoredEvent와 base_embedding을 SQLite에 저장하고 조회한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            _migrate_add_embedding_column(conn)

    # ------------------------------------------------------------------ #
    # 쓰기                                                                 #
    # ------------------------------------------------------------------ #

    def save(
        self,
        event: ScoredEvent,
        *,
        base_embedding: list[float] | None = None,
    ) -> None:
        """ScoredEvent를 저장한다. base_embedding이 있으면 함께 저장한다."""
        with self._connect() as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    event.query_id,
                    event.occurred_at.isoformat(),
                    event.translated_text,
                    event.embedding_model_id,
                    event.translation_model_id,
                    json.dumps(event.category_scores),
                    json.dumps(base_embedding) if base_embedding is not None else None,
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
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=days)
        ).isoformat()
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (cutoff,)).fetchall()
        return [_row_to_stored(row) for row in rows]

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


def _migrate_add_embedding_column(conn: sqlite3.Connection) -> None:
    """기존 DB에 base_embedding 컬럼이 없으면 추가한다."""
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(scored_events)").fetchall()
    }
    if "base_embedding" not in columns:
        conn.execute(_ADD_EMBEDDING_COLUMN_SQL)


def _row_to_stored(row: tuple) -> StoredScoredEvent:
    (
        query_id,
        occurred_at_str,
        translated_text,
        embedding_model_id,
        translation_model_id,
        category_scores_json,
        base_embedding_json,
    ) = row
    scored_event = ScoredEvent(
        query_id=query_id,
        occurred_at=datetime.fromisoformat(occurred_at_str),
        translated_text=translated_text,
        embedding_model_id=embedding_model_id,
        translation_model_id=translation_model_id,
        category_scores=json.loads(category_scores_json),
    )
    base_embedding = (
        json.loads(base_embedding_json) if base_embedding_json is not None else None
    )
    return StoredScoredEvent(scored_event=scored_event, base_embedding=base_embedding)
