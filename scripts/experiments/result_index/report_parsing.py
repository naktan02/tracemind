"""Result index report parsing helpers."""

from __future__ import annotations

import re
from typing import Any

RUN_TIMESTAMP_RE = re.compile(
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_(?P<hms>\d{6})"
)


def infer_created_at(run_id: str) -> str | None:
    match = RUN_TIMESTAMP_RE.search(run_id)
    if match is not None:
        hms = match.group("hms")
        return (
            f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
            f"T{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
        )
    compact_match = re.search(
        r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T"
        r"(?P<hms>\d{6})Z",
        run_id,
    )
    if compact_match is None:
        return None
    hms = compact_match.group("hms")
    return (
        f"{compact_match.group('year')}-{compact_match.group('month')}-"
        f"{compact_match.group('day')}"
        f"T{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
    )


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
