"""Query-domain PEFT SSL 모델 산출물 exporter."""

from __future__ import annotations

from typing import Any

import torch

from methods.adaptation.text_encoder_classifier.classifier_head_tensor_artifact import (
    save_classifier_head_state_tensor_artifact,
)
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
    """linear classifier head checkpoint를 tensor artifact로 저장한다."""

    save_classifier_head_state_tensor_artifact(
        path=classifier_path,
        classifier_state_dict=model.classifier.state_dict(),
        label_schema=categories,
    )


def save_legacy_classifier_head_pt_artifact(
    *,
    model: Any,
    categories: list[str],
    classifier_path,
) -> None:
    """구버전 `.pt` classifier head bundle을 저장한다."""

    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "categories": categories,
            "hidden_size": int(model.classifier.in_features),
        },
        classifier_path,
    )
