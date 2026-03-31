"""Prototype strategy 실험용 IO 유틸리티."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.run_artifacts import build_run_dir


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """JSONL 파일을 row 목록으로 읽는다."""
    resolved_path = Path(str(path))
    rows: list[dict[str, Any]] = []
    for line in resolved_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    """JSON 파일을 기록한다."""
    resolved_path = Path(str(path))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """JSONL 파일을 기록한다."""
    resolved_path = Path(str(path))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def resolve_output_dir(
    base_dir: str | Path,
    run_id: str,
    *,
    created_at: datetime | None = None,
) -> Path:
    """Hydra config의 base_dir를 run_id 실행 경로로 정규화한다."""
    return build_run_dir(base_dir, run_id=run_id, created_at=created_at)
