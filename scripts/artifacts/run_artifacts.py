"""실험 실행 산출물 경로 유틸리티."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_run_dir(
    base_dir: str | Path,
    *,
    run_id: str,
    created_at: datetime | None = None,
) -> Path:
    """`runs/<job>/<run_id>` 형태의 실행 경로를 만든다."""
    del created_at
    return Path(str(base_dir)) / run_id
