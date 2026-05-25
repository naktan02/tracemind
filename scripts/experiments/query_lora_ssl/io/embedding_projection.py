"""Query LoRA final representation projection artifact writer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.experiments.lora_classifier_projection import (
    write_lora_classifier_projection_artifacts,
)
from scripts.experiments.query_lora_ssl.io.artifact_paths import (
    QueryLoraRunArtifactPaths,
)


def write_query_lora_projection_artifacts(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any] | None,
    categories: list[str],
    device: str,
    paths: QueryLoraRunArtifactPaths,
    seed: int,
) -> dict[str, Any] | None:
    """최종 LoRA representation을 eval set별 2D projection artifact로 저장한다."""

    return write_lora_classifier_projection_artifacts(
        model=model,
        eval_loaders=eval_loaders,
        categories=categories,
        device=device,
        projection_dir=paths.projections_dir,
        seed=seed,
        schema_version="query_lora_projection_artifacts.v1",
    )
