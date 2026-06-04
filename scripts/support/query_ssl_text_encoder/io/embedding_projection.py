"""Query SSL final representation projection artifact writer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from methods.adaptation.peft_text_encoder.projection_artifacts import (
    write_peft_encoder_projection_artifacts,
)
from scripts.support.query_ssl_text_encoder.io.artifact_paths import (
    QueryPeftRunArtifactPaths,
)


def write_query_peft_projection_artifacts(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any] | None,
    categories: list[str],
    device: str,
    paths: QueryPeftRunArtifactPaths,
    seed: int,
) -> dict[str, Any] | None:
    """최종 PEFT encoder representation을 2D projection artifact로 저장한다."""

    return write_query_text_encoder_projection_artifacts(
        model=model,
        eval_loaders=eval_loaders,
        categories=categories,
        device=device,
        projection_dir=paths.projections_dir,
        seed=seed,
        schema_version="query_peft_projection_artifacts.v1",
        projection_space="final_peft_encoder_pooled_backbone_features",
        title_prefix="final PEFT encoder representation",
    )


def write_query_text_encoder_projection_artifacts(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any] | None,
    categories: list[str],
    device: str,
    projection_dir: Path,
    seed: int,
    schema_version: str,
    projection_space: str,
    title_prefix: str,
) -> dict[str, Any] | None:
    """최종 text encoder representation을 2D projection artifact로 저장한다."""

    return write_peft_encoder_projection_artifacts(
        model=model,
        eval_loaders=eval_loaders,
        categories=categories,
        device=device,
        projection_dir=projection_dir,
        seed=seed,
        schema_version=schema_version,
        projection_space=projection_space,
        title_prefix=title_prefix,
    )
