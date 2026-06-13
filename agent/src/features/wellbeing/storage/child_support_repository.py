"""아이용 지원 대화 로컬 SQLite 저장소.

이 저장소는 child-support 대화 원문과 응답을 agent 로컬에만 남긴다.
main_server나 shared artifact로 raw message를 승격하지 않는다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.src.config.paths import DEFAULT_AGENT_DATA_DIR

CHILD_SUPPORT_MESSAGE_V1 = "child_support_message.v1"

_DEFAULT_DB_PATH = DEFAULT_AGENT_DATA_DIR / "child_support_conversations.db"

_CREATE_CONVERSATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS child_support_conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    metadata        TEXT NOT NULL
);
"""

_CREATE_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS child_support_messages (
    message_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    role            TEXT NOT NULL,
    text            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    safety_level    TEXT,
    assistant_mode  TEXT,
    scope_status    TEXT,
    metadata        TEXT NOT NULL,
    FOREIGN KEY(conversation_id)
        REFERENCES child_support_conversations(conversation_id)
);
"""

_UPSERT_CONVERSATION_SQL = """
INSERT INTO child_support_conversations
    (conversation_id, created_at, updated_at, metadata)
VALUES (?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
    updated_at = excluded.updated_at;
"""

_INSERT_MESSAGE_SQL = """
INSERT OR REPLACE INTO child_support_messages
    (message_id, conversation_id, schema_version, role, text, created_at,
     safety_level, assistant_mode, scope_status, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT_MESSAGES_SQL = """
SELECT message_id, conversation_id, schema_version, role, text, created_at,
       safety_level, assistant_mode, scope_status, metadata
FROM child_support_messages
WHERE conversation_id = ?
ORDER BY created_at DESC
LIMIT ?;
"""

_COUNT_MESSAGES_SQL = """
SELECT COUNT(*)
FROM child_support_messages
WHERE conversation_id = ?;
"""


@dataclass(slots=True)
class ChildSupportMessageRecord:
    """agent 로컬 child-support message snapshot."""

    message_id: str
    conversation_id: str
    role: str
    text: str
    created_at: datetime
    safety_level: str | None = None
    assistant_mode: str | None = None
    scope_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CHILD_SUPPORT_MESSAGE_V1

    def __post_init__(self) -> None:
        if not self.message_id.strip():
            raise ValueError("message_id must not be empty.")
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be empty.")
        if self.role not in {"child", "assistant", "system"}:
            raise ValueError("role must be one of child, assistant, system.")
        if not self.text.strip():
            raise ValueError("text must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty.")


@dataclass(slots=True)
class ChildSupportConversationRepository:
    """아이 지원 대화 conversation/message를 로컬 SQLite에 저장한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_CONVERSATIONS_TABLE_SQL)
            conn.execute(_CREATE_MESSAGES_TABLE_SQL)

    def ensure_conversation(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """conversation row를 만들거나 updated_at만 갱신한다."""

        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                _UPSERT_CONVERSATION_SQL,
                (
                    conversation_id,
                    now,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def save_message(self, record: ChildSupportMessageRecord) -> None:
        """대화 메시지 snapshot을 저장한다."""

        self.ensure_conversation(record.conversation_id)
        with self._connect() as conn:
            conn.execute(
                _INSERT_MESSAGE_SQL,
                (
                    record.message_id,
                    record.conversation_id,
                    record.schema_version,
                    record.role,
                    record.text,
                    record.created_at.isoformat(),
                    record.safety_level,
                    record.assistant_mode,
                    record.scope_status,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )

    def get_recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 8,
    ) -> list[ChildSupportMessageRecord]:
        """conversation의 최근 메시지를 오래된 순서로 반환한다."""

        if limit <= 0:
            raise ValueError("limit must be positive.")
        with self._connect() as conn:
            rows = conn.execute(
                _SELECT_RECENT_MESSAGES_SQL, (conversation_id, limit)
            ).fetchall()
        records = [_row_to_message(row) for row in rows]
        return list(reversed(records))

    def count_messages(self, conversation_id: str) -> int:
        """conversation에 저장된 메시지 수를 반환한다."""

        with self._connect() as conn:
            row = conn.execute(_COUNT_MESSAGES_SQL, (conversation_id,)).fetchone()
        return int(row[0])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _row_to_message(row: sqlite3.Row) -> ChildSupportMessageRecord:
    metadata = json.loads(row["metadata"])
    if not isinstance(metadata, dict):
        metadata = {}
    return ChildSupportMessageRecord(
        message_id=str(row["message_id"]),
        conversation_id=str(row["conversation_id"]),
        schema_version=str(row["schema_version"]),
        role=str(row["role"]),
        text=str(row["text"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        safety_level=row["safety_level"],
        assistant_mode=row["assistant_mode"],
        scope_status=row["scope_status"],
        metadata=metadata,
    )
