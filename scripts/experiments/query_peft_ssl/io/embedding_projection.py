"""Query SSL final representation projection artifact writer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from methods.adaptation.peft_text_encoder.projection_artifacts import (
    write_peft_encoder_projection_artifacts,
)
from scripts.experiments.query_peft_ssl.io.artifact_paths import (
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

    return write_peft_encoder_projection_artifacts(
        model=model,
        eval_loaders=eval_loaders,
        categories=categories,
        device=device,
        projection_dir=paths.projections_dir,
        seed=seed,
        schema_version="query_peft_projection_artifacts.v1",
    )
