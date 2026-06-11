"""Query SSL family runner context 준비."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scripts.support.query_ssl_text_encoder.text_encoder_run_context import (
    TextEncoderRunContext,
    evaluate_text_encoder_run_context,
    prepare_text_encoder_run_context,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


@dataclass(slots=True)
class QuerySslRunContext:
    """Query SSL family runner가 공유하는 실행 컨텍스트."""

    cfg: Any
    effective_unlabeled_rows: list[LabeledQueryRow]
    effective_selection_set: str
    eval_set_map: dict[str, Path]
    effective_train_rows: list[LabeledQueryRow]
    eval_rows_by_name: dict[str, list[LabeledQueryRow]] | None
    categories: list[str]
    label_to_index: dict[str, int]
    training_device: str
    created_at: Any
    trainer_version: str
    model: Any
    tokenizer: Any
    backbone_summary: dict[str, Any]
    initial_checkpoint_manifest: dict[str, Any]
    train_loader: Any
    eval_loaders: dict[str, Any]
    selection_loader: Any


_QUERY_SSL_METHOD_IDENTITY_KEYS = {"name", "algorithm_name"}


def build_query_ssl_method_parameters(cfg) -> dict[str, object]:
    """algorithm별 method parameter를 Hydra config에서 plain dict로 추출한다."""

    method_payload = OmegaConf.to_container(cfg.query_ssl_method, resolve=True)
    if not isinstance(method_payload, dict):
        raise ValueError("query_ssl_method config must be a mapping.")
    return {
        str(key): value
        for key, value in method_payload.items()
        if str(key) not in _QUERY_SSL_METHOD_IDENTITY_KEYS
    }


def build_query_ssl_method_manifest(cfg) -> dict[str, object]:
    """Query SSL method config를 artifact manifest에 남길 canonical shape."""

    parameters = build_query_ssl_method_parameters(cfg)
    manifest: dict[str, object] = {
        "preset_name": str(cfg.query_ssl_method.name),
        "algorithm_name": str(cfg.query_ssl_method.algorithm_name),
        "parameters": parameters,
    }
    # 기존 report consumer를 깨지 않도록 parameter를 top-level에도 남긴다.
    manifest.update(parameters)
    return manifest


def prepare_query_ssl_run_context(
    cfg,
    *,
    train_rows: list[LabeledQueryRow] | None,
    unlabeled_rows: list[LabeledQueryRow] | None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    selection_set_name: str | None,
    categories_override: list[str] | tuple[str, ...] | None,
    model_builder: Callable[..., tuple[Any, Any, dict[str, Any]]],
    trainer_version_prefix: str,
    algorithm_name: str,
) -> QuerySslRunContext:
    """Query SSL family runner 공통 입력 정규화와 labeled/eval 준비를 수행한다."""

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

    base_context = prepare_text_encoder_run_context(
        cfg,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
        model_builder=model_builder,
        trainer_version_prefix=trainer_version_prefix,
    )

    return _build_query_ssl_run_context(
        base_context=base_context,
        effective_unlabeled_rows=effective_unlabeled_rows,
    )


def evaluate_query_ssl_run_context(
    *,
    model: Any,
    eval_loaders: Mapping[str, Any],
    categories: list[str],
    device: str,
) -> dict[str, Any]:
    """학습이 끝난 Query SSL 모델을 모든 eval set에서 평가한다."""

    return evaluate_text_encoder_run_context(
        model=model,
        eval_loaders=eval_loaders,
        categories=categories,
        device=device,
    )


def _build_query_ssl_run_context(
    *,
    base_context: TextEncoderRunContext,
    effective_unlabeled_rows: list[LabeledQueryRow],
) -> QuerySslRunContext:
    return QuerySslRunContext(
        cfg=base_context.cfg,
        effective_selection_set=base_context.effective_selection_set,
        eval_set_map=base_context.eval_set_map,
        effective_train_rows=base_context.effective_train_rows,
        eval_rows_by_name=base_context.eval_rows_by_name,
        effective_unlabeled_rows=effective_unlabeled_rows,
        categories=base_context.categories,
        label_to_index=base_context.label_to_index,
        training_device=base_context.training_device,
        created_at=base_context.created_at,
        trainer_version=base_context.trainer_version,
        model=base_context.model,
        tokenizer=base_context.tokenizer,
        backbone_summary=base_context.backbone_summary,
        initial_checkpoint_manifest=base_context.initial_checkpoint_manifest,
        train_loader=base_context.train_loader,
        eval_loaders=base_context.eval_loaders,
        selection_loader=base_context.selection_loader,
    )
