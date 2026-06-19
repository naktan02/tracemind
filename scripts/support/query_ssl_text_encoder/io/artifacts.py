"""Query-domain PEFT SSL 실험 산출물 저장 유틸리티."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.support.query_ssl_text_encoder.io.artifact_paths import (
    QueryPeftRunArtifactPaths,
    build_query_peft_run_artifact_paths,
)
from scripts.support.query_ssl_text_encoder.io.artifact_writer import (
    QueryTextEncoderRunArtifactWriter,
)
from scripts.support.query_ssl_text_encoder.io.embedding_projection import (
    write_query_peft_projection_artifacts,
)
from scripts.support.query_ssl_text_encoder.io.manifest_builder import (
    build_query_peft_eval_report,
    build_query_peft_run_manifest,
)
from scripts.support.query_ssl_text_encoder.io.model_artifact_exporter import (
    QueryPeftModelArtifactExporter,
)
from scripts.support.query_ssl_text_encoder.result_utils import (
    merge_results_with_best_and_final,
)


def write_run_artifacts(
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
    paths = build_query_peft_run_artifact_paths(
        cfg=cfg,
        trainer_version=trainer_version,
        created_at=created_at,
    )
    paths.ensure_directories()
    QueryPeftModelArtifactExporter().export(
        model=model,
        tokenizer=tokenizer,
        categories=categories,
        paths=paths,
    )
    projection_artifacts = write_query_peft_projection_artifacts(
        model=model,
        eval_loaders=_checkpoint_eval_loaders(
            eval_loaders=eval_loaders,
            selection_set=str(cfg.selection_set),
            checkpoint_name="best",
        ),
        categories=categories,
        device=training_device,
        paths=paths,
        seed=int(cfg.seed),
    )
    checkpoint_artifacts: dict[str, Any] = {
        "best": {
            "adapter_dir": str(paths.adapter_output_dir),
            "classifier_path": str(paths.classifier_path),
            "projection_artifacts": projection_artifacts,
        }
    }
    final_projection_artifacts: dict[str, Any] | None = None
    final_paths = _checkpoint_peft_paths(paths, "final")
    if final_model_state_dict is not None:
        final_paths.ensure_directories()
        with _loaded_model_state(model, final_model_state_dict):
            QueryPeftModelArtifactExporter().export(
                model=model,
                tokenizer=tokenizer,
                categories=categories,
                paths=final_paths,
            )
            final_projection_artifacts = write_query_peft_projection_artifacts(
                model=model,
                eval_loaders=_checkpoint_eval_loaders(
                    eval_loaders=eval_loaders,
                    selection_set=str(cfg.selection_set),
                    checkpoint_name="final",
                ),
                categories=categories,
                device=training_device,
                paths=final_paths,
                seed=int(cfg.seed),
            )
        checkpoint_artifacts["final"] = {
            "adapter_dir": str(final_paths.adapter_output_dir),
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
    manifest = build_query_peft_run_manifest(
        cfg=cfg,
        trainer_version=trainer_version,
        eval_set_map=eval_set_map,
        training_device=training_device,
        backbone_summary=backbone_summary,
        history=history,
        best_selection_report=best_selection_report,
        final_selection_report=final_selection_report,
        categories=categories,
        paths=paths,
        extra_manifest=effective_extra_manifest,
    )
    report = build_query_peft_eval_report(
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
                "final_adapter_dir": str(final_paths.adapter_output_dir),
                "final_classifier_path": str(final_paths.classifier_path),
                "final_projection_manifest": str(
                    final_paths.projections_dir / "projection_manifest.json"
                ),
            }
        )
    return outputs


def _checkpoint_peft_paths(
    paths: QueryPeftRunArtifactPaths,
    checkpoint_name: str,
) -> QueryPeftRunArtifactPaths:
    artifacts_dir = paths.output_dir / "artifacts" / checkpoint_name
    return replace(
        paths,
        adapter_output_dir=artifacts_dir / "adapter",
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
