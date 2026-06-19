"""Query-domain PEFT supervised baseline runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from methods.adaptation.peft_text_encoder.training.modeling import (
    build_model as build_query_peft_model,
)
from methods.adaptation.text_encoder_classifier.query_ssl_training import (
    train_classifier as train_query_peft_classifier,
)
from scripts.support.query_ssl_text_encoder.io.artifacts import write_run_artifacts
from scripts.support.query_ssl_text_encoder.runners.supervised_text_encoder import (
    run_supervised_text_encoder_baseline,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def run_supervised_peft_baseline(
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
    """PEFT baseline을 실행한다.

    기본 경로는 cfg가 가리키는 JSONL을 읽는다. query adaptation처럼 이미 메모리에
    조립된 labeled row가 있으면 override로 받아 JSONL bridge 없이 바로 학습한다.
    """

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
        train_classifier_func=train_query_peft_classifier,
        write_artifacts_func=write_run_artifacts,
        model_builder=build_query_peft_model,
        trainer_version_prefix="peft_clf",
    )
