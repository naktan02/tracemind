"""Agent runtime profile SQLite 저장소."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.local_agent_database import (
    DEFAULT_AGENT_LOCAL_DB_PATH,
    connect_agent_local_db,
)
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
)

CREATE_RUNTIME_PROFILE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_runtime_profiles (
    profile_id          TEXT NOT NULL,
    profile_revision    TEXT NOT NULL,
    payload_checksum    TEXT NOT NULL,
    source              TEXT NOT NULL,
    model_id            TEXT NOT NULL,
    model_revision      TEXT NOT NULL,
    runtime_family      TEXT NOT NULL,
    adapter_mechanism   TEXT,
    scorer_backend_name TEXT NOT NULL,
    embedding_backend   TEXT NOT NULL,
    embedding_model_id  TEXT NOT NULL,
    training_scope      TEXT NOT NULL,
    required_state_kind TEXT,
    payload_json        TEXT NOT NULL,
    received_at         TEXT NOT NULL,
    activated_at        TEXT,
    server_validated_at TEXT,
    server_base_url     TEXT,
    is_active           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (profile_id, profile_revision, payload_checksum)
);
"""

CREATE_ACTIVE_RUNTIME_PROFILE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runtime_profiles_single_active
ON agent_runtime_profiles (is_active)
WHERE is_active = 1;
"""

SELECT_ACTIVE_SQL = """
SELECT profile_id, profile_revision, payload_checksum, source, payload_json,
       received_at, activated_at, server_validated_at, server_base_url
FROM agent_runtime_profiles
WHERE is_active = 1
LIMIT 1;
"""

UPSERT_PROFILE_SQL = """
INSERT INTO agent_runtime_profiles
    (profile_id, profile_revision, payload_checksum, source, model_id,
     model_revision, runtime_family, adapter_mechanism, scorer_backend_name,
     embedding_backend, embedding_model_id, training_scope, required_state_kind,
     payload_json, received_at, activated_at, server_validated_at, server_base_url,
     is_active)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(profile_id, profile_revision, payload_checksum) DO UPDATE SET
    source = excluded.source,
    model_id = excluded.model_id,
    model_revision = excluded.model_revision,
    runtime_family = excluded.runtime_family,
    adapter_mechanism = excluded.adapter_mechanism,
    scorer_backend_name = excluded.scorer_backend_name,
    embedding_backend = excluded.embedding_backend,
    embedding_model_id = excluded.embedding_model_id,
    training_scope = excluded.training_scope,
    required_state_kind = excluded.required_state_kind,
    payload_json = excluded.payload_json,
    received_at = excluded.received_at,
    activated_at = COALESCE(excluded.activated_at, agent_runtime_profiles.activated_at),
    server_validated_at = COALESCE(
        excluded.server_validated_at,
        agent_runtime_profiles.server_validated_at
    ),
    server_base_url = COALESCE(
        excluded.server_base_url,
        agent_runtime_profiles.server_base_url
    ),
    is_active = excluded.is_active;
"""


@dataclass(frozen=True, slots=True)
class RuntimeProfileRecord:
    """agent-local runtime profile row."""

    profile: AgentRuntimeProfilePayload
    source: str
    received_at: datetime
    activated_at: datetime | None
    server_validated_at: datetime | None
    server_base_url: str | None


@dataclass(slots=True)
class RuntimeProfileRepository:
    """서버 runtime profile의 agent-local cache를 관리한다."""

    db_path: Path = DEFAULT_AGENT_LOCAL_DB_PATH

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            ensure_runtime_profile_schema(conn)

    def save_profile(
        self,
        profile: AgentRuntimeProfilePayload,
        *,
        source: str,
        activate: bool,
        received_at: datetime | None = None,
        activated_at: datetime | None = None,
        server_validated_at: datetime | None = None,
        server_base_url: str | None = None,
    ) -> RuntimeProfileRecord:
        """profile을 저장하고 필요하면 active profile로 전환한다."""

        effective_received_at = received_at or datetime.now(tz=timezone.utc)
        effective_activated_at = (
            activated_at or effective_received_at if activate else activated_at
        )
        with self._connect() as conn:
            if activate:
                conn.execute(
                    "UPDATE agent_runtime_profiles SET is_active = 0 "
                    "WHERE is_active = 1;"
                )
            conn.execute(
                UPSERT_PROFILE_SQL,
                (
                    profile.profile_id,
                    profile.profile_revision,
                    profile.payload_checksum,
                    _required_source(source),
                    profile.model_id,
                    profile.model_revision,
                    profile.runtime_family,
                    profile.adapter_mechanism,
                    profile.scorer_backend_name,
                    profile.embedding_backend,
                    profile.embedding_model_id,
                    profile.training_scope,
                    profile.required_state_kind,
                    profile.model_dump_json(),
                    effective_received_at.isoformat(),
                    _datetime_json(effective_activated_at),
                    _datetime_json(server_validated_at),
                    _optional_server_base_url(server_base_url),
                    1 if activate else 0,
                ),
            )
        return RuntimeProfileRecord(
            profile=profile,
            source=_required_source(source),
            received_at=effective_received_at,
            activated_at=effective_activated_at,
            server_validated_at=server_validated_at,
            server_base_url=_optional_server_base_url(server_base_url),
        )

    def load_active(self) -> RuntimeProfileRecord | None:
        """현재 active runtime profile을 반환한다."""

        with self._connect() as conn:
            row = conn.execute(SELECT_ACTIVE_SQL).fetchone()
        return None if row is None else _row_to_record(row)

    def mark_server_validated(
        self,
        *,
        profile_id: str,
        profile_revision: str,
        payload_checksum: str,
        validated_at: datetime | None = None,
        server_base_url: str | None = None,
    ) -> RuntimeProfileRecord:
        """서버 최신성 확인 시각을 저장한다."""

        effective_validated_at = validated_at or datetime.now(tz=timezone.utc)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE agent_runtime_profiles
                SET server_validated_at = ?,
                    server_base_url = COALESCE(?, server_base_url)
                WHERE profile_id = ?
                  AND profile_revision = ?
                  AND payload_checksum = ?;
                """,
                (
                    effective_validated_at.isoformat(),
                    _optional_server_base_url(server_base_url),
                    profile_id,
                    profile_revision,
                    payload_checksum,
                ),
            )
            if cursor.rowcount <= 0:
                raise FileNotFoundError("Runtime profile is not cached.")
            row = conn.execute(SELECT_ACTIVE_SQL).fetchone()
        if row is None:
            raise FileNotFoundError("No active runtime profile is cached.")
        return _row_to_record(row)

    def _connect(self) -> sqlite3.Connection:
        return connect_agent_local_db(self.db_path)


def ensure_runtime_profile_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_RUNTIME_PROFILE_TABLE_SQL)
    _ensure_column(
        conn,
        table_name="agent_runtime_profiles",
        column_name="server_base_url",
        column_sql="server_base_url TEXT",
    )
    conn.execute(CREATE_ACTIVE_RUNTIME_PROFILE_INDEX_SQL)


def _row_to_record(row: tuple[object, ...]) -> RuntimeProfileRecord:
    (
        _profile_id,
        _profile_revision,
        _payload_checksum,
        source,
        payload_json,
        received_at,
        activated_at,
        server_validated_at,
        server_base_url,
    ) = row
    return RuntimeProfileRecord(
        profile=AgentRuntimeProfilePayload.model_validate_json(str(payload_json)),
        source=str(source),
        received_at=datetime.fromisoformat(str(received_at)),
        activated_at=_optional_datetime(activated_at),
        server_validated_at=_optional_datetime(server_validated_at),
        server_base_url=_optional_server_base_url(server_base_url),
    )


def _ensure_column(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql};")


def _required_source(source: str) -> str:
    normalized = source.strip()
    if not normalized:
        raise ValueError("source must not be empty.")
    return normalized


def _datetime_json(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _optional_datetime(value: object) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


def _optional_server_base_url(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().rstrip("/")
    return normalized or None
