"""Query-domain PEFT SSL 모델 산출물 exporter."""

from __future__ import annotations

from typing import Any

import torch

from scripts.support.query_ssl_text_encoder.io.artifact_paths import (
    QueryPeftRunArtifactPaths,
)


class QueryPeftModelArtifactExporter:
    """adapter/tokenizer/classifier 파일 저장만 담당한다."""

    def export(
        self,
        *,
        model: Any,
        tokenizer: Any,
        categories: list[str],
        paths: QueryPeftRunArtifactPaths,
    ) -> None:
        model.backbone.save_pretrained(paths.adapter_output_dir)
        tokenizer.save_pretrained(paths.adapter_output_dir)
        save_classifier_head_artifact(
            model=model,
            categories=categories,
            classifier_path=paths.classifier_path,
        )


def save_classifier_head_artifact(
    *,
    model: Any,
    categories: list[str],
    classifier_path,
) -> None:
    """linear classifier head checkpoint를 공통 shape로 저장한다."""

    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "categories": categories,
            "hidden_size": int(model.classifier.in_features),
        },
        classifier_path,
    )
