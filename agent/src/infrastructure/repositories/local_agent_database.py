"""Agent-local SQLite database 공통 경로와 connection helper."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.src.config.paths import (
    DEFAULT_AGENT_DATA_DIR,
)

DEFAULT_AGENT_LOCAL_DB_PATH = DEFAULT_AGENT_DATA_DIR / "agent_local.db"


def connect_agent_local_db(db_path: Path) -> sqlite3.Connection:
    """FK enforcement를 켠 SQLite connection을 연다."""

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
