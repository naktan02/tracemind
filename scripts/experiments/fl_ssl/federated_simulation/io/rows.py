"""Federated simulation용 파일/시간 유틸리티."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


def load_jsonl_rows(path: str | Path) -> list[LabeledQueryRow]:
    """JSONL 파일을 row 목록으로 읽는다."""
    return load_labeled_query_rows(path)


def parse_created_at(value: str) -> datetime:
    """row의 created_at 문자열을 timezone-aware datetime으로 바꾼다."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
