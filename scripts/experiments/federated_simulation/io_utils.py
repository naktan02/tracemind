"""Federated simulation용 파일/시간 유틸리티."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """JSONL 파일을 row 목록으로 읽는다."""
    resolved_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line in resolved_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_created_at(value: str) -> datetime:
    """row의 created_at 문자열을 timezone-aware datetime으로 바꾼다."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
