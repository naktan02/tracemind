"""Full text encoder supervised control 산출물 저장 유틸리티."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.support.query_ssl_text_encoder.io.artifact_paths import (
    build_query_text_run_output_dir,
)
from scripts.support.query_ssl_text_encoder.io.artifact_writer import (
    QueryTextEncoderRunArtifactWriter,
)
from scripts.support.query_ssl_text_encoder.io.embedding_projection import (
    write_query_text_encoder_projection_artifacts,
)
from scripts.support.query_ssl_text_encoder.io.manifest_builder import (
    build_query_text_encoder_eval_report,
    build_query_text_encoder_run_manifest,
)
from scripts.support.query_ssl_text_encoder.io.model_artifact_exporter import (
    save_classifier_head_artifact,
)

FULL_TEXT_ENCODER_REPORT_SCHEMA_VERSION = "central_full_text_encoder_eval.v1"


@dataclass(frozen=True, slots=True)
class FullTextEncoderRunArtifactPaths:
    """Full text encoder supervised run 산출물 경로 묶음."""

    output_dir: Path
    model_output_dir: Path
    classifier_output_dir: Path
    classifier_path: Path
    classifier_manifest_path: Path
    report_path: Path
    logs_dir: Path
    projections_dir: Path

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.classifier_output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.projections_dir.mkdir(parents=True, exist_ok=True)

    def to_output_mapping(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "model_dir": str(self.model_output_dir),
            "classifier_path": str(self.classifier_path),
            "manifest": str(self.classifier_manifest_path),
            "report_json": str(self.report_path),
            "projection_manifest": str(
                self.projections_dir / "projection_manifest.json"
            ),
        }


def write_full_text_encoder_run_artifacts(
    *,
    cfg,
    trainer_version: str,
    created_at: datetime,
    model: Any,
    tokenizer: Any,
    categories: list[str],
    eval_set_map: dict[str, Path],
    training_device: str,
    backbone_summary: dict[str, Any],
    history: list[dict[str, Any]],
    best_selection_report: dict[str, Any],
    results: dict[str, Any],
    extra_manifest: Mapping[str, Any] | None = None,
    eval_loaders: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    paths = build_full_text_encoder_run_artifact_paths(
        cfg=cfg,
        trainer_version=trainer_version,
        created_at=created_at,
    )
    paths.ensure_directories()
    model.backbone.save_pretrained(paths.model_output_dir)
    tokenizer.save_pretrained(paths.model_output_dir)
    save_classifier_head_artifact(
        model=model,
        categories=categories,
        classifier_path=paths.classifier_path,
    )
    projection_artifacts = write_query_text_encoder_projection_artifacts(
        model=model,
        eval_loaders=eval_loaders,
        categories=categories,
        device=training_device,
        projection_dir=paths.projections_dir,
        seed=int(cfg.seed),
        schema_version="query_full_text_encoder_projection_artifacts.v1",
        projection_space="final_full_text_encoder_pooled_backbone_features",
        title_prefix="final full text encoder representation",
    )
    effective_extra_manifest = dict(extra_manifest or {})
    if projection_artifacts is not None:
        effective_extra_manifest["projection_artifacts"] = projection_artifacts
    manifest = build_query_text_encoder_run_manifest(
        cfg=cfg,
        trainer_version=trainer_version,
        eval_set_map=eval_set_map,
        training_device=training_device,
        backbone_summary=backbone_summary,
        history=history,
        best_selection_report=best_selection_report,
        categories=categories,
        model_artifact_fields={
            "model_dir": str(paths.model_output_dir),
            "classifier_path": str(paths.classifier_path),
        },
        extra_manifest=effective_extra_manifest,
    )
    report = build_query_text_encoder_eval_report(
        schema_version=FULL_TEXT_ENCODER_REPORT_SCHEMA_VERSION,
        trainer_version=trainer_version,
        manifest=manifest,
        results=results,
    )
    QueryTextEncoderRunArtifactWriter().write(
        paths=paths, manifest=manifest, report=report
    )
    return paths.to_output_mapping()


def build_full_text_encoder_run_artifact_paths(
    *,
    cfg: Any,
    trainer_version: str,
    created_at: datetime,
) -> FullTextEncoderRunArtifactPaths:
    output_dir = build_query_text_run_output_dir(
        cfg=cfg,
        trainer_version=trainer_version,
        created_at=created_at,
    )
    artifacts_dir = output_dir / "artifacts"
    model_output_dir = artifacts_dir / "model"
    classifier_output_dir = artifacts_dir
    return FullTextEncoderRunArtifactPaths(
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        classifier_output_dir=classifier_output_dir,
        classifier_path=classifier_output_dir / "classifier_head.pt",
        classifier_manifest_path=classifier_output_dir
        / "classifier_head.manifest.json",
        report_path=output_dir / "reports" / "report.json",
        logs_dir=output_dir / "logs",
        projections_dir=output_dir / "projections",
    )
