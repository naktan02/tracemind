"""Query-domain PEFT SSL 실험 산출물 저장 유틸리티."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.support.query_ssl_text_encoder.io.artifact_paths import (
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
        eval_loaders=eval_loaders,
        categories=categories,
        device=training_device,
        paths=paths,
        seed=int(cfg.seed),
    )
    final_results = merge_results_with_best_and_final(
        results=results,
        selection_set=str(cfg.selection_set),
        final_selection_report=(
            dict(final_selection_report)
            if isinstance(final_selection_report, dict)
            else None
        ),
    )
    effective_extra_manifest = dict(extra_manifest or {})
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
    return paths.to_output_mapping()
