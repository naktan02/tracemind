"""Wellbeing 저장소가 공유하는 SQLite 연결 유틸리티."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.src.config.paths import DEFAULT_AGENT_DATA_DIR

DEFAULT_WELLBEING_DB_PATH = DEFAULT_AGENT_DATA_DIR / "wellbeing_signal.db"


def connect_wellbeing_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)
