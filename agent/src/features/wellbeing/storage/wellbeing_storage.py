"""Wellbeing 저장소가 공유하는 SQLite 연결 유틸리티."""

from __future__ import annotations

import sqlite3
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WELLBEING_DB_PATH = AGENT_ROOT / "data" / "wellbeing_signal.db"


def connect_wellbeing_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)
