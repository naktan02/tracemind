"""FL simulation row-source validation for agent training inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def require_rows_supported_by_example_backend(
    *,
    rows: list[Mapping[str, Any]],
    backend_name: str,
) -> None:
    """선택된 example backend가 요구하는 source row shape를 검증한다."""

    from agent.src.services.training.backends.inputs.base import (
        WEAK_STRONG_PAIR_BACKEND_NAME,
    )

    if backend_name != WEAK_STRONG_PAIR_BACKEND_NAME:
        return
    _require_weak_strong_rows(rows)


def _require_weak_strong_rows(rows: list[Mapping[str, Any]]) -> None:
    for row in rows:
        if _has_legacy_weak_strong_fields(row) or _has_usb_view_fields(row):
            continue
        raise ValueError(
            "weak_strong_pair simulation requires each row to include both "
            "weak_text/strong_text or text plus aug_0/aug_1."
        )


def _has_legacy_weak_strong_fields(row: Mapping[str, Any]) -> bool:
    return bool(row.get("weak_text") and row.get("strong_text"))


def _has_usb_view_fields(row: Mapping[str, Any]) -> bool:
    return bool(row.get("text") and (row.get("aug_0") or row.get("aug_1")))
