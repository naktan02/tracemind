"""Query SSL family runner 공통 scaffolding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.src.infrastructure.runtime import resolve_runtime_device
from agent.src.services.training.query_adaptation.data import (
    build_dataloader,
    build_label_index,
)
from agent.src.services.training.query_adaptation.modeling import build_model
from agent.src.services.training.query_adaptation.training import (
    evaluate_classifier,
    set_seed,
)
from scripts.classification_report import (
    render_confusion_table,
    render_per_category_table,
)
from scripts.labeled_query_rows import LabeledQueryRow, load_labeled_query_rows

from ..initial_checkpoint import resolve_query_adaptation_initial_checkpoint


@dataclass(slots=True)
class QuerySslRunContext:
    """Query SSL family runner가 공유하는 실행 컨텍스트."""

    cfg: Any
    effective_selection_set: str
    eval_set_map: dict[str, Path]
    effective_train_rows: list[LabeledQueryRow]
    effective_unlabeled_rows: list[LabeledQueryRow]
    categories: list[str]
    label_to_index: dict[str, int]
    training_device: str
    created_at: datetime
    trainer_version: str
    model: Any
    tokenizer: Any
    backbone_summary: dict[str, Any]
    initial_checkpoint_manifest: dict[str, Any]
    train_loader: Any
    eval_loaders: dict[str, Any]
    selection_loader: Any


def build_query_ssl_method_manifest(cfg) -> dict[str, object]:
    """Query SSL method config를 artifact manifest에 남길 canonical shape."""

    return {
        "preset_name": str(cfg.query_ssl_method.name),
        "algorithm_name": str(cfg.query_ssl_method.algorithm_name),
        "temperature": float(cfg.query_ssl_method.temperature),
        "p_cutoff": float(cfg.query_ssl_method.p_cutoff),
        "hard_label": bool(cfg.query_ssl_method.hard_label),
        "lambda_u": float(cfg.query_ssl_method.lambda_u),
        "supervised_loss_weight": float(cfg.query_ssl_method.supervised_loss_weight),
        "unlabeled_batch_size": int(cfg.query_ssl_method.unlabeled_batch_size),
        "require_multiview": bool(cfg.query_ssl_method.require_multiview),
    }


def validate_multiview_rows(
    rows: Sequence[LabeledQueryRow],
    *,
    algorithm_name: str,
) -> None:
    """weak/strong pair가 필요한 consistency SSL 입력을 검증한다."""

    missing_query_ids = [
        str(row["query_id"])
        for row in rows
        if row.get("weak_text") is None or row.get("strong_text") is None
    ]
    if missing_query_ids:
        raise ValueError(
            f"{algorithm_name} requires each unlabeled row to include both "
            "weak_text and strong_text. Missing examples: "
            f"{missing_query_ids[:5]}."
        )


def prepare_query_ssl_run_context(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None,
    unlabeled_rows: list[LabeledQueryRow] | None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    selection_set_name: str | None,
    categories_override: list[str] | tuple[str, ...] | None,
    trainer_version_prefix: str,
    algorithm_name: str,
) -> QuerySslRunContext:
    """Query SSL family runner 공통 입력 정규화와 labeled/eval 준비를 수행한다."""

    effective_selection_set = str(selection_set_name or cfg.selection_set)
    eval_set_map = {name: Path(str(path)) for name, path in cfg.eval_sets.items()}
    if eval_rows_by_name is not None:
        for dataset_name in eval_rows_by_name:
            eval_set_map.setdefault(
                dataset_name, Path(f"in_memory/{dataset_name}.jsonl")
            )
    if effective_selection_set not in (
        eval_set_map if eval_rows_by_name is None else eval_rows_by_name
    ):
        raise ValueError(
            f"selection_set '{effective_selection_set}' is not included in eval_sets."
        )

    set_seed(int(cfg.seed))
    effective_train_rows = (
        load_labeled_query_rows(Path(str(cfg.train_jsonl)))
        if train_rows is None
        else list(train_rows)
    )
    if unlabeled_rows is None:
        if getattr(cfg, "unlabeled_jsonl", None) is None:
            raise ValueError(
                "unlabeled_jsonl is required unless unlabeled_rows is provided."
            )
        effective_unlabeled_rows = load_labeled_query_rows(
            Path(str(cfg.unlabeled_jsonl))
        )
    else:
        effective_unlabeled_rows = list(unlabeled_rows)
    if not effective_unlabeled_rows:
        raise ValueError(f"{algorithm_name} unlabeled_rows must not be empty.")

    effective_categories_override = categories_override
    if effective_categories_override is None:
        raw_fixed_categories = getattr(cfg, "fixed_categories", None)
        if raw_fixed_categories is not None:
            effective_categories_override = [
                str(category) for category in raw_fixed_categories
            ]

    if effective_categories_override is None:
        categories, label_to_index = build_label_index(effective_train_rows)
    else:
        categories = list(
            dict.fromkeys(str(category) for category in effective_categories_override)
        )
        if not categories:
            raise ValueError("categories_override must not be empty.")
        label_to_index = {
            str(category): index for index, category in enumerate(categories)
        }
        unknown_labels = sorted(
            {
                str(row["mapped_label_4"])
                for row in effective_train_rows
                if str(row["mapped_label_4"]) not in label_to_index
            }
        )
        if unknown_labels:
            raise ValueError(
                "Train rows include labels outside categories_override: "
                f"{unknown_labels}"
            )

    training_device = resolve_runtime_device(str(cfg.runtime.device))
    created_at = datetime.now(timezone.utc)
    trainer_version = cfg.trainer_version or created_at.strftime(
        f"{trainer_version_prefix}_%Y_%m_%d_%H%M%S"
    )
    resolved_initial_checkpoint = resolve_query_adaptation_initial_checkpoint(cfg)
    effective_cfg = resolved_initial_checkpoint.cfg

    model, tokenizer, backbone_summary = build_model(
        cfg=effective_cfg,
        categories=categories,
        device=training_device,
    )
    print(
        "trainable_params="
        f"{backbone_summary['parameter_counts']['trainable']} / "
        f"{backbone_summary['parameter_counts']['total']}",
        flush=True,
    )

    train_loader = build_dataloader(
        rows=effective_train_rows,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=int(effective_cfg.train_batch_size),
        max_length=int(effective_cfg.paper_backbone.max_length),
        task_prefix=str(effective_cfg.paper_backbone.task_prefix),
        shuffle=True,
    )

    eval_loaders: dict[str, Any] = {}
    eval_row_map = (
        None
        if eval_rows_by_name is None
        else {name: list(rows) for name, rows in eval_rows_by_name.items()}
    )
    eval_dataset_names = (
        eval_set_map.keys() if eval_row_map is None else eval_row_map.keys()
    )
    for dataset_name in eval_dataset_names:
        path = eval_set_map[dataset_name]
        rows = (
            load_labeled_query_rows(path)
            if eval_row_map is None
            else eval_row_map[dataset_name]
        )
        eval_loaders[dataset_name] = build_dataloader(
            rows=rows,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=int(effective_cfg.eval_batch_size),
            max_length=int(effective_cfg.paper_backbone.max_length),
            task_prefix=str(effective_cfg.paper_backbone.task_prefix),
            shuffle=False,
        )
        print(f"tokenized_eval_set={dataset_name} rows={len(rows)}", flush=True)

    return QuerySslRunContext(
        cfg=effective_cfg,
        effective_selection_set=effective_selection_set,
        eval_set_map=eval_set_map,
        effective_train_rows=effective_train_rows,
        effective_unlabeled_rows=effective_unlabeled_rows,
        categories=categories,
        label_to_index=label_to_index,
        training_device=training_device,
        created_at=created_at,
        trainer_version=trainer_version,
        model=model,
        tokenizer=tokenizer,
        backbone_summary=backbone_summary,
        initial_checkpoint_manifest=resolved_initial_checkpoint.extra_manifest,
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        selection_loader=eval_loaders[effective_selection_set],
    )


def evaluate_query_ssl_run_context(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any],
    categories: list[str],
    device: str,
) -> dict[str, Any]:
    """학습이 끝난 Query SSL 모델을 모든 eval set에서 평가한다."""

    results: dict[str, Any] = {}
    for dataset_name, dataloader in eval_loaders.items():
        report = evaluate_classifier(
            model=model,
            dataloader=dataloader,
            categories=categories,
            device=device,
        )
        results[dataset_name] = report
        print(
            f"[{dataset_name}] "
            f"accuracy_top_1={report['accuracy_top_1']:.4f} "
            f"rows={report['rows_total']} "
            f"mean_true_prob={report['mean_true_label_probability']:.4f} "
            f"mean_margin={report['mean_margin_top1_top2']:.4f}",
            flush=True,
        )
        print(render_confusion_table(report["confusion_matrix"]))
        print()
        print(
            render_per_category_table(
                report["per_category"],
                primary_metric_key="mean_true_label_probability",
                top_1_metric_key="mean_top_1_probability",
                primary_header="mean_true_prob",
                top_1_header="mean_top1_prob",
            )
        )
        print()
    return results
