"""FL simulation row-source validation for agent training inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from methods.adaptation.query_text_views.view_rows import (
    row_supports_weak_strong_pair,
)


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
    for row in rows:
        if row_supports_weak_strong_pair(row):
            continue
        raise ValueError(
            "weak_strong_pair simulation requires each row to include both "
            "weak_text/strong_text or text plus aug_0/aug_1."
        )
