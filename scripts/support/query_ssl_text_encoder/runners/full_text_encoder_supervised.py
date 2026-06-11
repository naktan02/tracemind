"""Full text encoder supervised baseline runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from methods.adaptation.full_text_encoder.training.modeling import (
    build_model as build_full_text_encoder_model,
)
from methods.adaptation.text_encoder_classifier.training import (
    train_classifier as train_text_encoder_classifier,
)
from scripts.support.query_ssl_text_encoder.io.full_text_encoder_artifacts import (
    write_full_text_encoder_run_artifacts,
)
from scripts.support.query_ssl_text_encoder.runners.supervised_text_encoder import (
    run_supervised_text_encoder_baseline,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def run_full_text_encoder_supervised_baseline(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None = None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None = None,
    selection_set_name: str | None = None,
    train_jsonl_ref: str | Path | None = None,
    eval_set_refs: Mapping[str, str | Path] | None = None,
    trainer_version_override: str | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
    categories_override: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Full text encoder + linear head supervised baseline을 실행한다."""

    return run_supervised_text_encoder_baseline(
        cfg=cfg,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        train_jsonl_ref=train_jsonl_ref,
        eval_set_refs=eval_set_refs,
        trainer_version_override=trainer_version_override,
        extra_manifest=extra_manifest,
        categories_override=categories_override,
        train_classifier_func=train_text_encoder_classifier,
        write_artifacts_func=write_full_text_encoder_run_artifacts,
        model_builder=build_full_text_encoder_model,
        trainer_version_prefix="full_text_encoder_clf",
    )
