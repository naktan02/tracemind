"""Query-domain PEFT SSL 산출물 경로 계산."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.support.artifacts.run_artifacts import build_run_dir


@dataclass(frozen=True, slots=True)
class QueryPeftRunArtifactPaths:
    """한 run에서 생성되는 산출물 경로 묶음."""

    output_dir: Path
    adapter_output_dir: Path
    classifier_output_dir: Path
    classifier_path: Path
    classifier_manifest_path: Path
    report_path: Path
    logs_dir: Path
    projections_dir: Path

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_output_dir.mkdir(parents=True, exist_ok=True)
        self.classifier_output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.projections_dir.mkdir(parents=True, exist_ok=True)

    def to_output_mapping(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "adapter_dir": str(self.adapter_output_dir),
            "classifier_path": str(self.classifier_path),
            "manifest": str(self.classifier_manifest_path),
            "report_json": str(self.report_path),
            "projection_manifest": str(
                self.projections_dir / "projection_manifest.json"
            ),
        }


def build_query_peft_run_artifact_paths(
    *,
    cfg: Any,
    trainer_version: str,
    created_at: datetime,
) -> QueryPeftRunArtifactPaths:
    output_dir = build_query_text_run_output_dir(
        cfg=cfg,
        trainer_version=trainer_version,
        created_at=created_at,
    )
    adapter_output_dir = Path(str(cfg.adapter_output_dir)) / trainer_version
    classifier_output_dir = Path(str(cfg.classifier_output_dir))
    return QueryPeftRunArtifactPaths(
        output_dir=output_dir,
        adapter_output_dir=adapter_output_dir,
        classifier_output_dir=classifier_output_dir,
        classifier_path=classifier_output_dir / f"{trainer_version}.pt",
        classifier_manifest_path=(
            classifier_output_dir / f"{trainer_version}.manifest.json"
        ),
        report_path=output_dir / "reports" / "report.json",
        logs_dir=output_dir / "logs",
        projections_dir=output_dir / "projections",
    )


def build_query_text_run_output_dir(
    *,
    cfg: Any,
    trainer_version: str,
    created_at: datetime,
) -> Path:
    """Query text encoder 계열 중앙 run output dir을 계산한다."""

    base_output_dir = Path(str(cfg.output_dir))
    run_group = _resolve_query_ssl_run_group(cfg)
    if run_group is not None:
        base_output_dir = base_output_dir / run_group
    return build_run_dir(
        base_output_dir,
        run_id=trainer_version,
        created_at=created_at,
    )


def _resolve_query_ssl_run_group(cfg: Any) -> str | None:
    if not bool(getattr(cfg, "group_by_query_ssl_method", False)):
        return None
    query_ssl_method = getattr(cfg, "query_ssl_method", None)
    if query_ssl_method is None:
        return None
    method_name = str(getattr(query_ssl_method, "name", "") or "").strip()
    if not method_name:
        return None
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_name)
    return safe_name.strip("._-") or None
