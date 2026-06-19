"""Full text encoder supervised control 산출물 저장 유틸리티."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
from scripts.support.query_ssl_text_encoder.result_utils import (
    merge_results_with_best_and_final,
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
    final_selection_report: dict[str, Any] | None,
    results: dict[str, Any],
    final_model_state_dict: Mapping[str, Any] | None = None,
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
        eval_loaders=_checkpoint_eval_loaders(
            eval_loaders=eval_loaders,
            selection_set=str(cfg.selection_set),
            checkpoint_name="best",
        ),
        categories=categories,
        device=training_device,
        projection_dir=paths.projections_dir,
        seed=int(cfg.seed),
        schema_version="query_full_text_encoder_projection_artifacts.v1",
        projection_space="final_full_text_encoder_pooled_backbone_features",
        title_prefix="final full text encoder representation",
    )
    checkpoint_artifacts: dict[str, Any] = {
        "best": {
            "model_dir": str(paths.model_output_dir),
            "classifier_path": str(paths.classifier_path),
            "projection_artifacts": projection_artifacts,
        }
    }
    final_paths = _checkpoint_full_paths(paths, "final")
    if final_model_state_dict is not None:
        final_paths.ensure_directories()
        with _loaded_model_state(model, final_model_state_dict):
            model.backbone.save_pretrained(final_paths.model_output_dir)
            tokenizer.save_pretrained(final_paths.model_output_dir)
            save_classifier_head_artifact(
                model=model,
                categories=categories,
                classifier_path=final_paths.classifier_path,
            )
            final_projection_artifacts = write_query_text_encoder_projection_artifacts(
                model=model,
                eval_loaders=_checkpoint_eval_loaders(
                    eval_loaders=eval_loaders,
                    selection_set=str(cfg.selection_set),
                    checkpoint_name="final",
                ),
                categories=categories,
                device=training_device,
                projection_dir=final_paths.projections_dir,
                seed=int(cfg.seed),
                schema_version="query_full_text_encoder_projection_artifacts.v1",
                projection_space="final_full_text_encoder_pooled_backbone_features",
                title_prefix="final full text encoder representation",
            )
        checkpoint_artifacts["final"] = {
            "model_dir": str(final_paths.model_output_dir),
            "classifier_path": str(final_paths.classifier_path),
            "projection_artifacts": final_projection_artifacts,
        }
    final_results = merge_results_with_best_and_final(
        results=results,
        selection_set=str(cfg.selection_set),
        final_selection_report=(
            dict(final_selection_report)
            if isinstance(final_selection_report, dict)
            else None
        ),
        include_selection_set_result=False,
    )
    effective_extra_manifest = dict(extra_manifest or {})
    effective_extra_manifest["checkpoint_artifacts"] = checkpoint_artifacts
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
        final_selection_report=final_selection_report,
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
        results=final_results,
    )
    QueryTextEncoderRunArtifactWriter().write(
        paths=paths, manifest=manifest, report=report
    )
    outputs = paths.to_output_mapping()
    if final_model_state_dict is not None:
        outputs.update(
            {
                "final_model_dir": str(final_paths.model_output_dir),
                "final_classifier_path": str(final_paths.classifier_path),
                "final_projection_manifest": str(
                    final_paths.projections_dir / "projection_manifest.json"
                ),
            }
        )
    return outputs


def _checkpoint_full_paths(
    paths: FullTextEncoderRunArtifactPaths,
    checkpoint_name: str,
) -> FullTextEncoderRunArtifactPaths:
    artifacts_dir = paths.output_dir / "artifacts" / checkpoint_name
    return replace(
        paths,
        model_output_dir=artifacts_dir / "model",
        classifier_output_dir=artifacts_dir,
        classifier_path=artifacts_dir / "classifier_head.safetensors",
        classifier_manifest_path=artifacts_dir / "classifier_head.manifest.json",
        projections_dir=paths.output_dir / "projections" / checkpoint_name,
    )


def _checkpoint_eval_loaders(
    *,
    eval_loaders: Mapping[str, Any] | None,
    selection_set: str,
    checkpoint_name: str,
) -> dict[str, Any] | None:
    if eval_loaders is None:
        return None
    loader = eval_loaders.get(selection_set)
    if loader is None and len(eval_loaders) == 1:
        loader = next(iter(eval_loaders.values()))
    if loader is None:
        return None
    return {checkpoint_name: loader}


@contextmanager
def _loaded_model_state(model: Any, state_dict: Mapping[str, Any]):
    current_state = {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in model.state_dict().items()
    }
    model.load_state_dict(dict(state_dict))
    try:
        yield
    finally:
        model.load_state_dict(current_state)


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
        classifier_path=classifier_output_dir / "classifier_head.safetensors",
        classifier_manifest_path=classifier_output_dir
        / "classifier_head.manifest.json",
        report_path=output_dir / "reports" / "report.json",
        logs_dir=output_dir / "logs",
        projections_dir=output_dir / "projections",
    )
