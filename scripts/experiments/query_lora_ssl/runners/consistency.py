"""Query SSL consistency family runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from methods.adaptation.lora_classifier.training import (
    train_query_ssl_classifier as train_query_ssl_lora_classifier,
)
from methods.adaptation.query_classifier_adaptation.data import (
    build_multiview_dataloader as build_query_lora_multiview_dataloader,
)
from methods.adaptation.query_classifier_adaptation.data import (
    build_weak_dataloader as build_query_lora_weak_dataloader,
)
from methods.ssl.base import QuerySslAlgorithmDescriptor
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from scripts.experiments.query_lora_ssl.io.artifacts import write_run_artifacts
from scripts.experiments.query_lora_ssl.query_ssl.augmentation import (
    PreparedQuerySslUnlabeledRows,
    build_query_ssl_augmenter_manifest,
    prepare_usb_multiview_unlabeled_rows,
    prepare_usb_weak_unlabeled_rows,
)
from scripts.experiments.query_lora_ssl.query_ssl.common import (
    QuerySslRunContext,
    build_query_ssl_method_manifest,
    build_query_ssl_method_parameters,
    evaluate_query_ssl_run_context,
    prepare_query_ssl_run_context,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


def run_query_ssl_lora_baseline(
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

    return run_consistency_query_ssl_lora_baseline(
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


def run_consistency_query_ssl_lora_baseline(
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
    prepared_unlabeled_rows = _prepare_unlabeled_rows(
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
    model, history, best_selection_report = train_query_ssl_lora_classifier(
        model=context.model,
        train_loader=context.train_loader,
        unlabeled_loader=unlabeled_loader,
        selection_loader=context.selection_loader,
        categories=context.categories,
        device=context.training_device,
        epochs=int(cfg.epochs),
        learning_rate=float(cfg.learning_rate),
        classifier_learning_rate=float(cfg.classifier_learning_rate),
        weight_decay=float(cfg.weight_decay),
        max_grad_norm=float(cfg.max_grad_norm),
        log_every_steps=int(cfg.log_every_steps),
        algorithm=algorithm,
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
    }
    effective_extra_manifest.update(context.initial_checkpoint_manifest)
    if getattr(cfg, "query_ssl_augmenter", None) is not None:
        effective_extra_manifest["query_ssl_augmenter"] = (
            build_query_ssl_augmenter_manifest(cfg)
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
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return outputs


def run_fixmatch_lora_baseline(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None = None,
    unlabeled_rows: list[LabeledQueryRow] | None = None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None = None,
    selection_set_name: str | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
    categories_override: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """USB FixMatch core를 Query SSL runner 위에서 실행한다."""

    descriptor = resolve_query_ssl_algorithm_descriptor(
        str(cfg.query_ssl_method.algorithm_name)
    )
    if descriptor.algorithm_name.strip().lower() != "fixmatch":
        raise ValueError(
            "run_fixmatch_lora_baseline requires "
            "query_ssl_method.algorithm_name=fixmatch."
        )
    return run_consistency_query_ssl_lora_baseline(
        cfg=cfg,
        descriptor=descriptor,
        train_rows=train_rows,
        unlabeled_rows=unlabeled_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        extra_manifest=extra_manifest,
        categories_override=categories_override,
    )


def run_pseudolabel_lora_baseline(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None = None,
    unlabeled_rows: list[LabeledQueryRow] | None = None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None = None,
    selection_set_name: str | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
    categories_override: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """USB PseudoLabel core를 Query SSL runner 위에서 실행한다."""

    descriptor = resolve_query_ssl_algorithm_descriptor(
        str(cfg.query_ssl_method.algorithm_name)
    )
    if descriptor.algorithm_name.strip().lower() != "pseudolabel":
        raise ValueError(
            "run_pseudolabel_lora_baseline requires "
            "query_ssl_method.algorithm_name=pseudolabel."
        )
    return run_consistency_query_ssl_lora_baseline(
        cfg=cfg,
        descriptor=descriptor,
        train_rows=train_rows,
        unlabeled_rows=unlabeled_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        extra_manifest=extra_manifest,
        categories_override=categories_override,
    )


def _prepare_unlabeled_rows(
    *,
    cfg,
    descriptor: QuerySslAlgorithmDescriptor,
    rows: list[LabeledQueryRow],
    source_jsonl: str | Path | None,
) -> PreparedQuerySslUnlabeledRows:
    if descriptor.required_views.view_builder_name == "usb_multiview":
        return prepare_usb_multiview_unlabeled_rows(
            cfg,
            rows=rows,
            source_jsonl=source_jsonl,
            algorithm_name=descriptor.display_name,
        )
    if descriptor.required_views.view_builder_name == "usb_weak":
        return prepare_usb_weak_unlabeled_rows(
            rows=rows,
            algorithm_name=descriptor.display_name,
        )
    raise ValueError(
        "Unsupported Query SSL view builder: "
        f"{descriptor.required_views.view_builder_name}."
    )


def _build_unlabeled_loader(
    *,
    cfg,
    descriptor: QuerySslAlgorithmDescriptor,
    context: QuerySslRunContext,
):
    if descriptor.required_views.view_builder_name == "usb_multiview":
        return build_query_lora_multiview_dataloader(
            rows=context.effective_unlabeled_rows,
            tokenizer=context.tokenizer,
            batch_size=int(cfg.query_ssl_method.unlabeled_batch_size),
            max_length=int(cfg.paper_backbone.max_length),
            task_prefix=str(cfg.paper_backbone.task_prefix),
            shuffle=True,
        )
    if descriptor.required_views.view_builder_name == "usb_weak":
        return build_query_lora_weak_dataloader(
            rows=context.effective_unlabeled_rows,
            tokenizer=context.tokenizer,
            batch_size=int(cfg.query_ssl_method.unlabeled_batch_size),
            max_length=int(cfg.paper_backbone.max_length),
            task_prefix=str(cfg.paper_backbone.task_prefix),
            shuffle=True,
        )
    raise ValueError(
        "Unsupported Query SSL view builder: "
        f"{descriptor.required_views.view_builder_name}."
    )


def _build_trainer_version_prefix(descriptor: QuerySslAlgorithmDescriptor) -> str:
    return f"lora_{descriptor.algorithm_name.strip().lower()}"
