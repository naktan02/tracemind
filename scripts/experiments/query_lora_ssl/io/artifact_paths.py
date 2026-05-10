"""Query-domain LoRA SSL 산출물 경로 계산."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.artifacts.run_artifacts import build_run_dir


@dataclass(frozen=True, slots=True)
class QueryLoraRunArtifactPaths:
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


def build_query_lora_run_artifact_paths(
    *,
    cfg: Any,
    trainer_version: str,
    created_at: datetime,
) -> QueryLoraRunArtifactPaths:
    output_dir = build_run_dir(
        cfg.output_dir,
        run_id=trainer_version,
        created_at=created_at,
    )
    adapter_output_dir = Path(str(cfg.adapter_output_dir)) / trainer_version
    classifier_output_dir = Path(str(cfg.classifier_output_dir))
    return QueryLoraRunArtifactPaths(
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
