"""Shared helpers for FL SSL report verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def object_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def object_sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, list | tuple) else ()


def nested_or_flat_value(
    payload: Mapping[str, object],
    namespace: str,
    key: str,
) -> object:
    flat_value = payload.get(f"{namespace}.{key}")
    if flat_value is not None:
        return flat_value
    return object_mapping(payload.get(namespace)).get(key)


def resolve_report_path(summary_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return summary_path.parent / path


def optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def expect_equal(
    errors: list[str],
    field: str,
    observed: object,
    expected: object,
) -> None:
    if expected is not None and observed != expected:
        errors.append(f"{field} expected {expected!r}, got {observed!r}.")


def expect_float_equal(
    errors: list[str],
    field: str,
    observed: object,
    expected: float | None,
) -> None:
    if expected is None:
        return
    if observed is None:
        errors.append(f"{field} expected {expected!r}, got None.")
        return
    if float(observed) != expected:
        errors.append(f"{field} expected {expected!r}, got {observed!r}.")


def expect_contains(
    errors: list[str],
    field: str,
    observed: object,
    expected_substring: str | None,
) -> None:
    if expected_substring is None:
        return
    if not isinstance(observed, str) or expected_substring not in observed:
        errors.append(
            f"{field} expected to contain {expected_substring!r}, got {observed!r}."
        )
