"""Query-domain LoRA SSL JSON 산출물 writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.experiments.query_peft_ssl.io.artifact_paths import (
    QueryLoraRunArtifactPaths,
)


class QueryLoraRunArtifactWriter:
    """manifest/report JSON 쓰기만 담당한다."""

    def write(
        self,
        *,
        paths: QueryLoraRunArtifactPaths,
        manifest: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        self.write_json(path=paths.classifier_manifest_path, payload=manifest)
        self.write_json(path=paths.report_path, payload=report)

    def write_json(self, *, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
