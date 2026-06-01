"""중앙 supervised text encoder baseline 공통 runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scripts.support.query_ssl_text_encoder.runtime_context import (
    evaluate_supervised_text_encoder_run_context,
    prepare_supervised_text_encoder_run_context,
)
from scripts.support.query_ssl_text_encoder.runtime_metrics import (
    run_with_training_runtime_metrics,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def run_supervised_text_encoder_baseline(
    *,
    cfg,
    train_rows: list[LabeledQueryRow] | None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    selection_set_name: str | None,
    train_jsonl_ref: str | Path | None,
    eval_set_refs: Mapping[str, str | Path] | None,
    trainer_version_override: str | None,
    extra_manifest: Mapping[str, Any] | None,
    categories_override: list[str] | tuple[str, ...] | None,
    train_classifier_func: Callable[
        ..., tuple[Any, list[dict[str, Any]], dict[str, Any]]
    ],
    write_artifacts_func: Callable[..., dict[str, str]],
    model_builder: Callable[..., tuple[Any, Any, dict[str, Any]]],
    trainer_version_prefix: str,
) -> dict[str, str]:
    """중앙 supervised text encoder baseline 공통 실행 흐름."""

    context = prepare_supervised_text_encoder_run_context(
        cfg,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
        model_builder=model_builder,
        train_jsonl_ref=train_jsonl_ref,
        eval_set_refs=eval_set_refs,
        trainer_version_override=trainer_version_override,
        trainer_version_prefix=trainer_version_prefix,
    )
    max_train_steps = _resolve_max_train_steps(context.cfg)
    (
        (model, history, best_selection_report),
        runtime_metrics,
    ) = run_with_training_runtime_metrics(
        lambda: train_classifier_func(
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

    results = evaluate_supervised_text_encoder_run_context(
        model=model,
        eval_loaders=context.eval_loaders,
        categories=context.categories,
        device=context.training_device,
    )

    effective_extra_manifest = dict(context.initial_checkpoint_manifest)
    effective_extra_manifest["runtime_metrics"] = runtime_metrics
    if extra_manifest:
        effective_extra_manifest.update(dict(extra_manifest))

    outputs = write_artifacts_func(
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
