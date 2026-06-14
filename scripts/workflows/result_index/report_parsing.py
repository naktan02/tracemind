"""Result index report parsing helpers."""

from __future__ import annotations

import json
import re
from typing import Any

RUN_TIMESTAMP_RE = re.compile(
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_(?P<hms>\d{6})"
)
LABEL_BUDGET_PATTERNS = (
    re.compile(r"(?:^|_)labels_pc(?P<count>\d+)(?:_|$)"),
    re.compile(r"(?:^|_)labels-pc(?P<count>\d+)(?:_|$)"),
    re.compile(r"(?:^|/)labeled(?P<count>\d+)_per_class(?:_|/)"),
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


def infer_label_budget_count_from_texts(*values: Any) -> int | None:
    """path/slug에 들어간 per-class label budget 표기를 정규화한다."""

    for value in values:
        text = str(value or "")
        if not text:
            continue
        for pattern in LABEL_BUDGET_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                return optional_int(match.group("count"))
    return None


def label_budget_name_from_count(count: int | None) -> str | None:
    return f"pc{count}" if count is not None else None


def json_object_snapshot(
    value: dict[str, Any],
    *,
    excluded_keys: set[str] | None = None,
) -> str | None:
    """가변 parameter object를 stable JSON 문자열로 보존한다."""

    excluded = set() if excluded_keys is None else excluded_keys
    payload = {
        str(key): item
        for key, item in value.items()
        if key not in excluded and item is not None
    }
    if not payload:
        return None
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
