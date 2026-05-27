"""Query SSL consistency family runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from methods.adaptation.peft_text_classifier.training.loops import (
    train_query_ssl_classifier as train_query_ssl_peft_classifier,
)
from methods.adaptation.query_text_views.query_ssl_views import (
    build_query_ssl_unlabeled_dataloader,
)
from methods.ssl.base import QuerySslAlgorithmDescriptor
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from scripts.experiments.query_peft_ssl.io.artifacts import write_run_artifacts
from scripts.experiments.query_peft_ssl.query_ssl.augmentation import (
    build_query_ssl_augmenter_manifest,
)
from scripts.experiments.query_peft_ssl.query_ssl.common import (
    QuerySslRunContext,
    build_query_ssl_method_manifest,
    build_query_ssl_method_parameters,
    evaluate_query_ssl_run_context,
    prepare_query_ssl_run_context,
)
from scripts.experiments.query_peft_ssl.query_ssl.view_preparation import (
    prepare_query_ssl_unlabeled_rows,
)
from scripts.experiments.query_peft_ssl.runtime_metrics import (
    run_with_training_runtime_metrics,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


def run_query_ssl_peft_baseline(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None = None,
    unlabeled_rows: list[LabeledQueryRow] | None = None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None = None,
    selection_set_name: str | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
    categories_override: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """query_ssl_method.algorithm_name에 맞는 중앙 Query SSL baseline을 실행한다."""

    return run_consistency_query_ssl_peft_baseline(
        cfg=cfg,
        descriptor=resolve_query_ssl_algorithm_descriptor(
            str(cfg.query_ssl_method.algorithm_name)
        ),
        train_rows=train_rows,
        unlabeled_rows=unlabeled_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        extra_manifest=extra_manifest,
        categories_override=categories_override,
    )


def run_consistency_query_ssl_peft_baseline(
    cfg,
    *,
    descriptor: QuerySslAlgorithmDescriptor,
    train_rows: list[LabeledQueryRow] | None = None,
    unlabeled_rows: list[LabeledQueryRow] | None = None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None = None,
    selection_set_name: str | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
    categories_override: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Query SSL baseline을 공통 scaffolding으로 실행한다."""

    if unlabeled_rows is None:
        if getattr(cfg, "unlabeled_jsonl", None) is None:
            raise ValueError(
                "unlabeled_jsonl is required unless unlabeled_rows is provided."
            )
        raw_unlabeled_rows = load_labeled_query_rows(Path(str(cfg.unlabeled_jsonl)))
    else:
        raw_unlabeled_rows = list(unlabeled_rows)
    prepared_unlabeled_rows = prepare_query_ssl_unlabeled_rows(
        cfg=cfg,
        descriptor=descriptor,
        rows=raw_unlabeled_rows,
        source_jsonl=getattr(cfg, "unlabeled_jsonl", None),
    )

    context = prepare_query_ssl_run_context(
        cfg=cfg,
        train_rows=train_rows,
        unlabeled_rows=prepared_unlabeled_rows.rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
        trainer_version_prefix=_build_trainer_version_prefix(descriptor),
        algorithm_name=descriptor.display_name,
    )
    unlabeled_loader = _build_unlabeled_loader(
        cfg=cfg,
        descriptor=descriptor,
        context=context,
    )
    algorithm = descriptor.build_algorithm(build_query_ssl_method_parameters(cfg))
    max_train_steps = _resolve_max_train_steps(cfg)
    (
        (model, history, best_selection_report),
        runtime_metrics,
    ) = run_with_training_runtime_metrics(
        lambda: train_query_ssl_peft_classifier(
            model=context.model,
            train_loader=context.train_loader,
            unlabeled_loader=unlabeled_loader,
            selection_loader=context.selection_loader,
            categories=context.categories,
            device=context.training_device,
            epochs=int(cfg.epochs),
            max_train_steps=max_train_steps,
            learning_rate=float(cfg.learning_rate),
            classifier_learning_rate=float(cfg.classifier_learning_rate),
            weight_decay=float(cfg.weight_decay),
            max_grad_norm=float(cfg.max_grad_norm),
            log_every_steps=int(cfg.log_every_steps),
            algorithm=algorithm,
            resume_checkpoint_path=getattr(cfg, "resume_checkpoint_path", None),
            resume_checkpoint_output_dir=getattr(
                cfg,
                "resume_checkpoint_output_dir",
                None,
            ),
            resume_checkpoint_every_epochs=int(
                getattr(cfg, "resume_checkpoint_every_epochs", 0)
            ),
        ),
        training_example_count=_estimate_query_ssl_training_example_count(
            cfg=cfg,
            algorithm=algorithm,
            max_train_steps=max_train_steps,
            train_row_count=len(context.effective_train_rows),
            unlabeled_row_count=len(context.effective_unlabeled_rows),
        ),
        parameter_counts=context.backbone_summary["parameter_counts"],
        device=context.training_device,
    )
    results = evaluate_query_ssl_run_context(
        model=model,
        eval_loaders=context.eval_loaders,
        categories=context.categories,
        device=context.training_device,
    )

    effective_extra_manifest: dict[str, Any] = {
        "unlabeled_jsonl": None
        if getattr(cfg, "unlabeled_jsonl", None) is None
        else str(cfg.unlabeled_jsonl),
        "unlabeled_row_count": len(context.effective_unlabeled_rows),
        "query_ssl_method": build_query_ssl_method_manifest(cfg),
        "query_ssl_resume": _build_query_ssl_resume_manifest(cfg),
        "runtime_metrics": runtime_metrics,
    }
    effective_extra_manifest.update(context.initial_checkpoint_manifest)
    if (
        descriptor.required_views.view_builder_name == "usb_multiview"
        and getattr(cfg, "query_ssl_augmenter", None) is not None
    ):
        effective_extra_manifest["query_ssl_augmenter"] = (
            build_query_ssl_augmenter_manifest(cfg)
        )
        effective_extra_manifest["query_ssl_strong_view_policy"] = (
            _build_query_ssl_strong_view_policy_manifest(cfg)
        )
    effective_extra_manifest.update(prepared_unlabeled_rows.build_run_manifest())
    if extra_manifest is not None:
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


def _estimate_query_ssl_training_example_count(
    *,
    cfg: Any,
    algorithm: Any,
    max_train_steps: int | None,
    train_row_count: int,
    unlabeled_row_count: int,
) -> int:
    if max_train_steps is None:
        return (
            (train_row_count if algorithm.uses_labeled_batches else 0)
            + unlabeled_row_count
        ) * int(cfg.epochs)
    labeled_batch_size = (
        int(cfg.train_batch_size) if algorithm.uses_labeled_batches else 0
    )
    unlabeled_batch_size = int(cfg.query_ssl_method.unlabeled_batch_size)
    return int(max_train_steps) * (labeled_batch_size + unlabeled_batch_size)


def _build_unlabeled_loader(
    *,
    cfg,
    descriptor: QuerySslAlgorithmDescriptor,
    context: QuerySslRunContext,
):
    return build_query_ssl_unlabeled_dataloader(
        rows=context.effective_unlabeled_rows,
        tokenizer=context.tokenizer,
        batch_size=int(cfg.query_ssl_method.unlabeled_batch_size),
        max_length=int(cfg.paper_backbone.max_length),
        task_prefix=str(cfg.paper_backbone.task_prefix),
        shuffle=True,
        view_builder_name=descriptor.required_views.view_builder_name,
        strong_view_policy=_resolve_strong_view_policy(cfg),
    )


def _resolve_strong_view_policy(cfg: Any) -> str:
    raw_policy = getattr(cfg, "query_ssl_strong_view_policy", None)
    if raw_policy is None:
        return "first_aug"
    return str(getattr(raw_policy, "policy", raw_policy))


def _build_query_ssl_strong_view_policy_manifest(cfg: Any) -> dict[str, object]:
    return {"policy": _resolve_strong_view_policy(cfg)}


def _build_query_ssl_resume_manifest(cfg: Any) -> dict[str, object]:
    resume_checkpoint_path = getattr(cfg, "resume_checkpoint_path", None)
    resume_checkpoint_output_dir = getattr(cfg, "resume_checkpoint_output_dir", None)
    return {
        "resume_checkpoint_path": None
        if resume_checkpoint_path is None or not str(resume_checkpoint_path).strip()
        else str(resume_checkpoint_path),
        "checkpoint_output_dir": None
        if resume_checkpoint_output_dir is None
        or not str(resume_checkpoint_output_dir).strip()
        else str(resume_checkpoint_output_dir),
        "checkpoint_every_epochs": int(
            getattr(cfg, "resume_checkpoint_every_epochs", 0)
        ),
    }


def _build_trainer_version_prefix(descriptor: QuerySslAlgorithmDescriptor) -> str:
    return f"lora_{descriptor.algorithm_name.strip().lower()}"
