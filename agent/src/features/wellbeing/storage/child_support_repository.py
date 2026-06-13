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

_CREATE_PROACTIVE_CLAIMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS child_support_proactive_prompt_claims (
    prompt_id       TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    message_id      TEXT NOT NULL,
    claimed_at      TEXT NOT NULL,
    prompt_text     TEXT NOT NULL,
    safety_level    TEXT,
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

_INSERT_PROACTIVE_CLAIM_SQL = """
INSERT INTO child_support_proactive_prompt_claims
    (prompt_id, conversation_id, message_id, claimed_at, prompt_text,
     safety_level, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(prompt_id) DO NOTHING;
"""

_SELECT_PROACTIVE_CLAIM_SQL = """
SELECT prompt_id, conversation_id, message_id, claimed_at, prompt_text,
       safety_level, metadata
FROM child_support_proactive_prompt_claims
WHERE prompt_id = ?
LIMIT 1;
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


@dataclass(frozen=True, slots=True)
class ChildSupportProactivePromptClaimRecord:
    """선제 발화가 실제 대화로 claim된 기록."""

    prompt_id: str
    conversation_id: str
    message_id: str
    claimed_at: datetime
    prompt_text: str
    safety_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChildSupportConversationRepository:
    """아이 지원 대화 conversation/message를 로컬 SQLite에 저장한다."""

    db_path: Path = _DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_CONVERSATIONS_TABLE_SQL)
            conn.execute(_CREATE_MESSAGES_TABLE_SQL)
            conn.execute(_CREATE_PROACTIVE_CLAIMS_TABLE_SQL)

    def ensure_conversation(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """conversation row를 만들거나 updated_at만 갱신한다."""

        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            _ensure_conversation(
                conn,
                conversation_id,
                metadata=metadata,
                now_iso=now,
            )

    def save_message(self, record: ChildSupportMessageRecord) -> None:
        """대화 메시지 snapshot을 저장한다."""

        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            _ensure_conversation(
                conn,
                record.conversation_id,
                metadata=None,
                now_iso=now,
            )
            conn.execute(_INSERT_MESSAGE_SQL, _message_params(record))

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

    def get_proactive_prompt_claim(
        self,
        prompt_id: str,
    ) -> ChildSupportProactivePromptClaimRecord | None:
        """prompt_id로 기존 proactive claim을 조회한다."""

        with self._connect() as conn:
            row = conn.execute(_SELECT_PROACTIVE_CLAIM_SQL, (prompt_id,)).fetchone()
        return None if row is None else _row_to_proactive_claim(row)

    def claim_proactive_prompt(
        self,
        *,
        message: ChildSupportMessageRecord,
        claim: ChildSupportProactivePromptClaimRecord,
    ) -> ChildSupportProactivePromptClaimRecord:
        """첫 assistant message와 proactive claim을 같은 transaction에 저장한다."""

        with self._connect() as conn:
            now = datetime.now(tz=timezone.utc).isoformat()
            _ensure_conversation(
                conn,
                message.conversation_id,
                metadata=None,
                now_iso=now,
            )
            conn.execute(_INSERT_MESSAGE_SQL, _message_params(message))
            cursor = conn.execute(
                _INSERT_PROACTIVE_CLAIM_SQL,
                _proactive_claim_params(claim),
            )
            if cursor.rowcount > 0:
                return claim
            conn.rollback()
            row = conn.execute(
                _SELECT_PROACTIVE_CLAIM_SQL,
                (claim.prompt_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("proactive prompt claim conflict could not be loaded.")
        return _row_to_proactive_claim(row)

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


def _ensure_conversation(
    conn: sqlite3.Connection,
    conversation_id: str,
    *,
    metadata: dict[str, Any] | None,
    now_iso: str,
) -> None:
    conn.execute(
        _UPSERT_CONVERSATION_SQL,
        (
            conversation_id,
            now_iso,
            now_iso,
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )


def _message_params(record: ChildSupportMessageRecord) -> tuple[object, ...]:
    return (
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
    )


def _proactive_claim_params(
    record: ChildSupportProactivePromptClaimRecord,
) -> tuple[object, ...]:
    return (
        record.prompt_id,
        record.conversation_id,
        record.message_id,
        record.claimed_at.isoformat(),
        record.prompt_text,
        record.safety_level,
        json.dumps(record.metadata, ensure_ascii=False),
    )


def _row_to_proactive_claim(
    row: sqlite3.Row,
) -> ChildSupportProactivePromptClaimRecord:
    metadata = json.loads(row["metadata"])
    if not isinstance(metadata, dict):
        metadata = {}
    return ChildSupportProactivePromptClaimRecord(
        prompt_id=str(row["prompt_id"]),
        conversation_id=str(row["conversation_id"]),
        message_id=str(row["message_id"]),
        claimed_at=datetime.fromisoformat(str(row["claimed_at"])),
        prompt_text=str(row["prompt_text"]),
        safety_level=row["safety_level"],
        metadata=metadata,
    )
