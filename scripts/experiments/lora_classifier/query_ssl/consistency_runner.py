"""Query SSL consistency family runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.src.services.training.query_adaptation.algorithms.registry import (
    build_query_ssl_algorithm,
)
from agent.src.services.training.query_adaptation.data import (
    build_multiview_dataloader,
)
from agent.src.services.training.query_adaptation.training import (
    train_query_ssl_classifier,
)
from scripts.labeled_query_rows import LabeledQueryRow, load_labeled_query_rows

from ..artifacts import write_run_artifacts
from .augmentation import (
    PreparedQuerySslUnlabeledRows,
    build_query_ssl_augmenter_manifest,
    prepare_usb_multiview_unlabeled_rows,
)
from .common import (
    QuerySslRunContext,
    build_query_ssl_method_manifest,
    build_query_ssl_method_parameters,
    evaluate_query_ssl_run_context,
    prepare_query_ssl_run_context,
)


@dataclass(frozen=True, slots=True)
class QuerySslAlgorithmAdapter:
    """Query SSL algorithm별 scripts wiring."""

    algorithm_name: str
    trainer_version_prefix: str
    prepare_unlabeled_rows: Callable[
        [list[LabeledQueryRow], str | Path | None],
        PreparedQuerySslUnlabeledRows,
    ]
    build_unlabeled_loader: Callable[[QuerySslRunContext], Any]
    train: Callable[
        [QuerySslRunContext, Any],
        tuple[Any, list[dict[str, Any]], dict[str, Any]],
    ]


QuerySslAlgorithmAdapterFactory = Callable[[Any], QuerySslAlgorithmAdapter]
ConsistencyMethodAdapter = QuerySslAlgorithmAdapter

_QUERY_SSL_ALGORITHM_ADAPTER_REGISTRY: dict[str, QuerySslAlgorithmAdapterFactory] = {}


def register_query_ssl_algorithm_adapter(
    *algorithm_names: str,
    factory: QuerySslAlgorithmAdapterFactory,
) -> None:
    """query_ssl_method.algorithm_name별 scripts adapter를 등록한다."""

    for algorithm_name in algorithm_names:
        _QUERY_SSL_ALGORITHM_ADAPTER_REGISTRY[algorithm_name.strip().lower()] = factory


def build_query_ssl_algorithm_adapter(cfg) -> QuerySslAlgorithmAdapter:
    """Hydra query_ssl_method에서 scripts adapter를 선택한다."""

    algorithm_name = str(cfg.query_ssl_method.algorithm_name)
    factory = _QUERY_SSL_ALGORITHM_ADAPTER_REGISTRY.get(algorithm_name.strip().lower())
    if factory is None:
        raise ValueError(f"Unsupported query SSL algorithm adapter: {algorithm_name}.")
    return factory(cfg)


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
        adapter=build_query_ssl_algorithm_adapter(cfg),
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
    adapter: QuerySslAlgorithmAdapter,
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
    prepared_unlabeled_rows = adapter.prepare_unlabeled_rows(
        raw_unlabeled_rows,
        getattr(cfg, "unlabeled_jsonl", None),
    )

    context = prepare_query_ssl_run_context(
        cfg=cfg,
        train_rows=train_rows,
        unlabeled_rows=prepared_unlabeled_rows.rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
        trainer_version_prefix=adapter.trainer_version_prefix,
        algorithm_name=adapter.algorithm_name,
    )
    unlabeled_loader = adapter.build_unlabeled_loader(context)
    model, history, best_selection_report = adapter.train(context, unlabeled_loader)
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

    return run_consistency_query_ssl_lora_baseline(
        cfg=cfg,
        adapter=_build_fixmatch_adapter(cfg),
        train_rows=train_rows,
        unlabeled_rows=unlabeled_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        extra_manifest=extra_manifest,
        categories_override=categories_override,
    )


def _build_fixmatch_adapter(cfg) -> ConsistencyMethodAdapter:
    algorithm_name = str(cfg.query_ssl_method.algorithm_name)
    if algorithm_name.strip().lower() != "fixmatch":
        raise ValueError(
            "run_fixmatch_lora_baseline requires "
            "query_ssl_method.algorithm_name=fixmatch."
        )
    algorithm = build_query_ssl_algorithm(
        algorithm_name="fixmatch",
        parameters=build_query_ssl_method_parameters(cfg),
    )

    def _prepare_unlabeled_rows(
        rows: list[LabeledQueryRow],
        source_jsonl: str | Path | None,
    ) -> PreparedQuerySslUnlabeledRows:
        return prepare_usb_multiview_unlabeled_rows(
            cfg,
            rows=rows,
            source_jsonl=source_jsonl,
            algorithm_name="FixMatch",
        )

    def _build_unlabeled_loader(context: QuerySslRunContext):
        return build_multiview_dataloader(
            rows=context.effective_unlabeled_rows,
            tokenizer=context.tokenizer,
            batch_size=int(cfg.query_ssl_method.unlabeled_batch_size),
            max_length=int(cfg.paper_backbone.max_length),
            task_prefix=str(cfg.paper_backbone.task_prefix),
            shuffle=True,
        )

    def _train(
        context: QuerySslRunContext,
        unlabeled_loader,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        return train_query_ssl_classifier(
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

    return QuerySslAlgorithmAdapter(
        algorithm_name="FixMatch",
        trainer_version_prefix="lora_fixmatch",
        prepare_unlabeled_rows=_prepare_unlabeled_rows,
        build_unlabeled_loader=_build_unlabeled_loader,
        train=_train,
    )


register_query_ssl_algorithm_adapter("fixmatch", factory=_build_fixmatch_adapter)
