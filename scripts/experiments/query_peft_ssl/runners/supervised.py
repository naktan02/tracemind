"""Query-domain LoRA supervised baseline runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from methods.adaptation.peft_text_classifier.training.loops import (
    train_classifier as train_query_lora_classifier,
)
from scripts.experiments.query_peft_ssl.harness.common import (
    evaluate_supervised_lora_run_context,
    prepare_supervised_lora_run_context,
)
from scripts.experiments.query_peft_ssl.io.artifacts import write_run_artifacts
from scripts.experiments.query_peft_ssl.runtime_metrics import (
    run_with_training_runtime_metrics,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def run_supervised_lora_baseline(
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
    """LoRA baseline을 실행한다.

    기본 경로는 cfg가 가리키는 JSONL을 읽는다. query adaptation처럼 이미 메모리에
    조립된 labeled row가 있으면 override로 받아 JSONL bridge 없이 바로 학습한다.
    """

    context = prepare_supervised_lora_run_context(
        cfg,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
        train_jsonl_ref=train_jsonl_ref,
        eval_set_refs=eval_set_refs,
        trainer_version_override=trainer_version_override,
    )
    max_train_steps = _resolve_max_train_steps(context.cfg)
    (
        (model, history, best_selection_report),
        runtime_metrics,
    ) = run_with_training_runtime_metrics(
        lambda: train_query_lora_classifier(
            model=context.model,
            train_loader=context.train_loader,
            selection_loader=context.selection_loader,
            categories=context.categories,
            device=context.training_device,
            epochs=int(context.cfg.epochs),
            max_train_steps=max_train_steps,
            learning_rate=float(context.cfg.learning_rate),
            classifier_learning_rate=float(context.cfg.classifier_learning_rate),
            weight_decay=float(context.cfg.weight_decay),
            max_grad_norm=float(context.cfg.max_grad_norm),
            log_every_steps=int(context.cfg.log_every_steps),
        ),
        training_example_count=_estimate_supervised_training_example_count(
            cfg=context.cfg,
            max_train_steps=max_train_steps,
            train_row_count=len(context.effective_train_rows),
        ),
        parameter_counts=context.backbone_summary["parameter_counts"],
        device=context.training_device,
    )

    results = evaluate_supervised_lora_run_context(
        model=model,
        eval_loaders=context.eval_loaders,
        categories=context.categories,
        device=context.training_device,
    )

    effective_extra_manifest = dict(context.initial_checkpoint_manifest)
    effective_extra_manifest["runtime_metrics"] = runtime_metrics
    if extra_manifest:
        effective_extra_manifest.update(dict(extra_manifest))

    outputs = write_run_artifacts(
        cfg=context.cfg,
        trainer_version=context.trainer_version,
        created_at=context.created_at,
        model=model,
        tokenizer=context.tokenizer,
        categories=context.categories,
        eval_set_map=context.eval_set_map,
        training_device=context.training_device,
        backbone_summary=context.backbone_summary,
        history=history,
        best_selection_report=best_selection_report,
        results=results,
        extra_manifest=effective_extra_manifest,
        eval_loaders=context.eval_loaders,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return outputs


def _resolve_max_train_steps(cfg: Any) -> int | None:
    raw_value = getattr(cfg, "max_train_steps", None)
    if raw_value is None:
        return None
    return int(raw_value)


def _estimate_supervised_training_example_count(
    *,
    cfg: Any,
    max_train_steps: int | None,
    train_row_count: int,
) -> int:
    if max_train_steps is None:
        return train_row_count * int(cfg.epochs)
    return int(max_train_steps) * int(cfg.train_batch_size)
