"""Query-domain text encoder JSON 산출물 writer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class QueryTextEncoderRunArtifactWriter:
    """manifest/report JSON 쓰기만 담당한다."""

    def write(
        self,
        *,
        paths,
        manifest: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        self.write_json(path=paths.classifier_manifest_path, payload=manifest)
        self.write_json(path=paths.report_path, payload=report)

    def write_json(self, *, path: Path, payload: dict[str, Any]) -> None:
        write_json_artifact(path=path, payload=payload)


def write_json_artifact(*, path: Path, payload: Mapping[str, Any]) -> None:
    """JSON object artifact를 repo 표준 포맷으로 쓴다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl_artifact(
    *,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """JSONL row artifact를 repo 표준 포맷으로 쓴다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=True) + "\n")
