"""CapturedTextEvent agent-local SQLite 저장소."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.src.contracts.captured_text_contracts import (
    CAPTURED_TEXT_EVENT_V1,
    CapturedTextEventPayload,
)

_DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "captured_text.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS captured_text_events (
    event_id          TEXT PRIMARY KEY,
    schema_version    TEXT NOT NULL,
    occurred_at       TEXT NOT NULL,
    received_at       TEXT NOT NULL,
    text              TEXT NOT NULL,
    locale            TEXT NOT NULL,
    source_type       TEXT NOT NULL,
    surface_type      TEXT NOT NULL,
    page_url          TEXT,
    page_title        TEXT,
    collector_version TEXT,
    metadata          TEXT NOT NULL
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO captured_text_events
    (event_id, schema_version, occurred_at, received_at, text, locale,
     source_type, surface_type, page_url, page_title, collector_version, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_ONE_SQL = """
SELECT event_id, schema_version, occurred_at, received_at, text, locale,
       source_type, surface_type, page_url, page_title, collector_version, metadata
FROM captured_text_events
WHERE event_id = ?;
"""

_SELECT_RECENT_SQL = """
SELECT event_id, schema_version, occurred_at, received_at, text, locale,
       source_type, surface_type, page_url, page_title, collector_version, metadata
FROM captured_text_events
ORDER BY occurred_at DESC
LIMIT ?;
"""

_COUNT_SQL = "SELECT COUNT(*) FROM captured_text_events;"


@dataclass(slots=True)
class CapturedTextRecord:
    """agent 로컬 captured text raw event snapshot."""

    event_id: str
    occurred_at: datetime
    received_at: datetime
    text: str
    locale: str
    source_type: str
    surface_type: str
    page_url: str | None = None
    page_title: str | None = None
    collector_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CAPTURED_TEXT_EVENT_V1

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")
        if not self.text.strip():
            raise ValueError("text must not be empty.")
        if not self.locale.strip():
            raise ValueError("locale must not be empty.")
        if not self.source_type.strip():
            raise ValueError("source_type must not be empty.")
        if not self.surface_type.strip():
            raise ValueError("surface_type must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty.")


def captured_text_record_from_payload(
    payload: CapturedTextEventPayload,
    *,
    received_at: datetime | None = None,
) -> CapturedTextRecord:
    """CapturedTextEventPayload를 저장소 record로 정규화한다."""

    return CapturedTextRecord(
        event_id=payload.event_id,
        schema_version=payload.schema_version,
        occurred_at=payload.occurred_at,
        received_at=received_at or datetime.now(tz=timezone.utc),
        text=payload.text,
        locale=payload.locale,
        source_type=payload.source_type.value,
        surface_type=payload.surface_type.value,
        page_url=payload.page_url,
        page_title=payload.page_title,
        collector_version=payload.collector_version,
        metadata=dict(payload.metadata),
    )


@dataclass(slots=True)
class CapturedTextRepository:
    """CapturedTextRecord를 SQLite에 저장하고 최소 조회를 제공한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def save(self, record: CapturedTextRecord) -> None:
        """event_id 기준으로 raw captured text event를 저장한다."""

        with self._connect() as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    record.event_id,
                    record.schema_version,
                    record.occurred_at.isoformat(),
                    record.received_at.isoformat(),
                    record.text,
                    record.locale,
                    record.source_type,
                    record.surface_type,
                    record.page_url,
                    record.page_title,
                    record.collector_version,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )

    def get(self, event_id: str) -> CapturedTextRecord | None:
        """단일 event_id의 저장 레코드를 반환한다."""

        with self._connect() as conn:
            row = conn.execute(_SELECT_ONE_SQL, (event_id,)).fetchone()
        return None if row is None else _row_to_record(row)

    def get_recent(self, *, limit: int = 50) -> list[CapturedTextRecord]:
        """최근 captured text event를 최신순으로 반환한다."""

        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (limit,)).fetchall()
        return [_row_to_record(row) for row in rows]

    def count(self) -> int:
        """저장된 captured text event 수를 반환한다."""

        with self._connect() as conn:
            return conn.execute(_COUNT_SQL).fetchone()[0]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


def _row_to_record(row: tuple[Any, ...]) -> CapturedTextRecord:
    (
        event_id,
        schema_version,
        occurred_at,
        received_at,
        text,
        locale,
        source_type,
        surface_type,
        page_url,
        page_title,
        collector_version,
        metadata_json,
    ) = row
    metadata = json.loads(str(metadata_json))
    if not isinstance(metadata, dict):
        metadata = {}
    return CapturedTextRecord(
        event_id=str(event_id),
        schema_version=str(schema_version),
        occurred_at=datetime.fromisoformat(str(occurred_at)),
        received_at=datetime.fromisoformat(str(received_at)),
        text=str(text),
        locale=str(locale),
        source_type=str(source_type),
        surface_type=str(surface_type),
        page_url=None if page_url is None else str(page_url),
        page_title=None if page_title is None else str(page_title),
        collector_version=None if collector_version is None else str(collector_version),
        metadata=metadata,
    )
