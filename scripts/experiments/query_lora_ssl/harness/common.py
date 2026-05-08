"""Query-domain LoRA SSL runner 공통 scaffolding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from methods.adaptation.lora_classifier.modeling import (
    build_model as build_query_lora_model,
)
from methods.adaptation.lora_classifier.training import (
    evaluate_classifier as evaluate_query_lora_classifier,
)
from methods.adaptation.lora_classifier.training import (
    set_seed as set_query_lora_seed,
)
from methods.adaptation.query_classifier_adaptation.data import (
    build_dataloader as build_query_lora_dataloader,
)
from methods.adaptation.query_classifier_adaptation.data import (
    build_label_index as build_query_lora_label_index,
)
from scripts.experiments.query_lora_ssl.config.initial_checkpoint import (
    resolve_query_adaptation_initial_checkpoint,
)
from scripts.io.labeled_query_rows import LabeledQueryRow, load_labeled_query_rows
from scripts.runtime_adapters.embedding_runtime import resolve_runtime_device_name
from shared.src.domain.services.classification_report import (
    render_confusion_table,
    render_per_category_table,
)


@dataclass(slots=True)
class SupervisedLoraRunContext:
    """Supervised LoRA runner가 공유하는 실행 컨텍스트."""

    cfg: Any
    effective_selection_set: str
    eval_set_map: dict[str, Path]
    effective_train_rows: list[LabeledQueryRow]
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


LoraLabeledRunContext = SupervisedLoraRunContext


def prepare_supervised_lora_run_context(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    selection_set_name: str | None,
    categories_override: list[str] | tuple[str, ...] | None,
    train_jsonl_ref: str | Path | None = None,
    eval_set_refs: Mapping[str, str | Path] | None = None,
    trainer_version_override: str | None = None,
    trainer_version_prefix: str = "lora_clf",
) -> SupervisedLoraRunContext:
    """Supervised LoRA runner 공통 입력 정규화와 dataloader 준비를 수행한다."""

    effective_selection_set = str(selection_set_name or cfg.selection_set)
    effective_train_jsonl_ref = str(
        cfg.train_jsonl if train_jsonl_ref is None else train_jsonl_ref
    )
    eval_set_map = _resolve_eval_set_map(
        cfg=cfg,
        eval_rows_by_name=eval_rows_by_name,
        eval_set_refs=eval_set_refs,
    )
    if effective_selection_set not in (
        eval_set_map if eval_rows_by_name is None else eval_rows_by_name
    ):
        raise ValueError(
            f"selection_set '{effective_selection_set}' is not included in eval_sets."
        )

    set_query_lora_seed(int(cfg.seed))
    effective_train_rows = (
        load_labeled_query_rows(Path(effective_train_jsonl_ref))
        if train_rows is None
        else list(train_rows)
    )
    categories, label_to_index = _resolve_categories(
        rows=effective_train_rows,
        categories_override=categories_override,
        raw_fixed_categories=getattr(cfg, "fixed_categories", None),
    )

    training_device = resolve_runtime_device_name(str(cfg.runtime.device))
    created_at = datetime.now(timezone.utc)
    trainer_version = (
        trainer_version_override
        or cfg.trainer_version
        or created_at.strftime(f"{trainer_version_prefix}_%Y_%m_%d_%H%M%S")
    )
    resolved_initial_checkpoint = resolve_query_adaptation_initial_checkpoint(cfg)
    effective_cfg = _clone_cfg_with_run_data_refs(
        cfg=resolved_initial_checkpoint.cfg,
        train_jsonl_ref=effective_train_jsonl_ref,
        eval_set_map=eval_set_map,
        selection_set_name=effective_selection_set,
    )

    model, tokenizer, backbone_summary = build_query_lora_model(
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

    train_loader = build_query_lora_dataloader(
        rows=effective_train_rows,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=int(effective_cfg.train_batch_size),
        max_length=int(effective_cfg.paper_backbone.max_length),
        task_prefix=str(effective_cfg.paper_backbone.task_prefix),
        shuffle=True,
    )
    eval_loaders = build_eval_loaders(
        cfg=effective_cfg,
        eval_set_map=eval_set_map,
        eval_rows_by_name=eval_rows_by_name,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
    )

    return SupervisedLoraRunContext(
        cfg=effective_cfg,
        effective_selection_set=effective_selection_set,
        eval_set_map=eval_set_map,
        effective_train_rows=effective_train_rows,
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


def prepare_labeled_lora_run_context(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    selection_set_name: str | None,
    categories_override: list[str] | tuple[str, ...] | None,
    train_jsonl_ref: str | Path | None = None,
    eval_set_refs: Mapping[str, str | Path] | None = None,
    trainer_version_override: str | None = None,
    trainer_version_prefix: str = "lora_clf",
) -> LoraLabeledRunContext:
    """Labeled LoRA family runner 공통 입력 정규화와 dataloader 준비를 수행한다."""

    return prepare_supervised_lora_run_context(
        cfg,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
        train_jsonl_ref=train_jsonl_ref,
        eval_set_refs=eval_set_refs,
        trainer_version_override=trainer_version_override,
        trainer_version_prefix=trainer_version_prefix,
    )


def build_eval_loaders(
    *,
    cfg,
    eval_set_map: Mapping[str, Path],
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    label_to_index: dict[str, int],
    tokenizer: Any,
) -> dict[str, Any]:
    """Eval set들을 dataloader로 변환한다."""

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
        eval_loaders[dataset_name] = build_query_lora_dataloader(
            rows=rows,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=int(cfg.eval_batch_size),
            max_length=int(cfg.paper_backbone.max_length),
            task_prefix=str(cfg.paper_backbone.task_prefix),
            shuffle=False,
        )
        print(f"tokenized_eval_set={dataset_name} rows={len(rows)}", flush=True)
    return eval_loaders


def evaluate_supervised_lora_run_context(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any],
    categories: list[str],
    device: str,
) -> dict[str, Any]:
    """학습이 끝난 supervised LoRA 모델을 모든 eval set에서 평가한다."""

    results: dict[str, Any] = {}
    for dataset_name, dataloader in eval_loaders.items():
        report = evaluate_query_lora_classifier(
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


def evaluate_lora_run_context(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any],
    categories: list[str],
    device: str,
) -> dict[str, Any]:
    """Labeled LoRA family 모델을 모든 eval set에서 평가한다."""

    return evaluate_supervised_lora_run_context(
        model=model,
        eval_loaders=eval_loaders,
        categories=categories,
        device=device,
    )


def _resolve_eval_set_map(
    *,
    cfg,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    eval_set_refs: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    eval_set_map = {name: Path(str(path)) for name, path in cfg.eval_sets.items()}
    if eval_set_refs is not None:
        for dataset_name, path in eval_set_refs.items():
            eval_set_map[str(dataset_name)] = Path(str(path))
    if eval_rows_by_name is not None:
        for dataset_name in eval_rows_by_name:
            eval_set_map.setdefault(
                dataset_name, Path(f"in_memory/{dataset_name}.jsonl")
            )
    return eval_set_map


def _clone_cfg_with_run_data_refs(
    *,
    cfg,
    train_jsonl_ref: str,
    eval_set_map: Mapping[str, Path],
    selection_set_name: str,
):
    cloned_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cloned_cfg.train_jsonl = train_jsonl_ref
    cloned_cfg.eval_sets = {name: str(path) for name, path in eval_set_map.items()}
    cloned_cfg.selection_set = selection_set_name
    return cloned_cfg


def _resolve_categories(
    *,
    rows: list[LabeledQueryRow],
    categories_override: list[str] | tuple[str, ...] | None,
    raw_fixed_categories: Any,
) -> tuple[list[str], dict[str, int]]:
    effective_categories_override = categories_override
    if effective_categories_override is None and raw_fixed_categories is not None:
        effective_categories_override = [
            str(category) for category in raw_fixed_categories
        ]

    if effective_categories_override is None:
        return build_query_lora_label_index(rows)

    categories = list(
        dict.fromkeys(str(category) for category in effective_categories_override)
    )
    if not categories:
        raise ValueError("categories_override must not be empty.")
    label_to_index = {str(category): index for index, category in enumerate(categories)}
    unknown_labels = sorted(
        {
            str(row["mapped_label_4"])
            for row in rows
            if str(row["mapped_label_4"]) not in label_to_index
        }
    )
    if unknown_labels:
        raise ValueError(
            f"Train rows include labels outside categories_override: {unknown_labels}"
        )
    return categories, label_to_index
