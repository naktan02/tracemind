"""Full text encoder Query SSL local training session."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from methods.adaptation.query_text_views.data import build_dataloader
from methods.adaptation.query_text_views.local_training_budget import (
    QuerySslLocalStepPlan,
    build_query_ssl_local_step_plan,
)
from methods.adaptation.query_text_views.query_ssl_views import (
    build_query_ssl_unlabeled_dataloader,
)
from methods.adaptation.query_text_views.view_rows import (
    validate_query_ssl_unlabeled_views,
)
from methods.adaptation.text_encoder_classifier.modeling import (
    TextEncoderWithLinearHead,
)
from methods.adaptation.text_encoder_classifier.pseudo_label_diagnostics import (
    build_final_snapshot_pseudo_label_quality,
    resolve_fixed_pseudo_label_diagnostic_threshold,
    tokenization_cache_namespace,
)
from methods.adaptation.text_encoder_classifier.query_ssl_session import (
    CentralQuerySslTextEncoderSessionRequest,
    QuerySslTextEncoderLocalTrainerOptions,
)
from methods.adaptation.text_encoder_classifier.query_ssl_training import (
    train_query_ssl_classifier,
)
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(frozen=True, slots=True)
class QuerySslFullTextEncoderLocalSslResult:
    """중앙 full text encoder Query SSL local 학습 결과."""

    model: TextEncoderWithLinearHead
    tokenizer: Any
    algorithm: Any
    effective_labeled_rows: tuple[LabeledQueryRow, ...]
    effective_unlabeled_rows: tuple[LabeledQueryRow, ...]
    effective_labels: tuple[str, ...]
    selection_rows: tuple[LabeledQueryRow, ...]
    local_step_plan: QuerySslLocalStepPlan
    history: tuple[dict[str, Any], ...]
    best_selection_report: Mapping[str, Any]
    final_model_state_dict: Mapping[str, Any] | None
    pseudo_label_quality: PseudoLabelQualitySummary
    diagnostic_client_metrics: Mapping[str, float]
    tokenization_cache_namespace: str


@dataclass(frozen=True, slots=True)
class _QuerySslFullTextEncoderLocalRuntime:
    """Full Query SSL local 학습 loop에 필요한 준비된 runtime."""

    model: TextEncoderWithLinearHead
    tokenizer: Any
    algorithm: Any
    effective_labeled_rows: tuple[LabeledQueryRow, ...]
    effective_unlabeled_rows: tuple[LabeledQueryRow, ...]
    effective_labels: tuple[str, ...]
    selection_rows: tuple[LabeledQueryRow, ...]
    train_loader: Any
    selection_loader: Any
    unlabeled_loader: Any
    step_plan: QuerySslLocalStepPlan
    backbone_config: Any
    tokenization_cache_namespace: str


def run_query_ssl_full_text_encoder_local_session(
    request: CentralQuerySslTextEncoderSessionRequest,
) -> QuerySslFullTextEncoderLocalSslResult:
    """공통 중앙 request surface로 full text encoder Query SSL 학습을 실행한다."""

    trainer_options = (
        QuerySslTextEncoderLocalTrainerOptions()
        if request.trainer_options is None
        else request.trainer_options
    )
    runtime = _prepare_query_ssl_local_runtime(request)
    final_model_state_dict: dict[str, Any] | None = None

    def capture_final_model_state(model: TextEncoderWithLinearHead) -> None:
        nonlocal final_model_state_dict
        final_model_state_dict = _clone_model_state_dict(model)

    before_restore_best = (
        capture_final_model_state if trainer_options.capture_final_model_state else None
    )
    model, history, best_selection_report = train_query_ssl_classifier(
        model=runtime.model,
        train_loader=runtime.train_loader,
        unlabeled_loader=runtime.unlabeled_loader,
        selection_loader=runtime.selection_loader,
        categories=list(runtime.effective_labels),
        device=request.trainer_runtime_config.device,
        epochs=int(request.training_task.local_epochs),
        max_train_steps=runtime.step_plan.total_steps,
        learning_rate=float(request.training_task.learning_rate),
        classifier_learning_rate=(
            float(request.training_task.learning_rate)
            if trainer_options.classifier_learning_rate is None
            else float(trainer_options.classifier_learning_rate)
        ),
        weight_decay=float(trainer_options.weight_decay),
        proximal_mu=0.0,
        max_grad_norm=(
            0.0
            if request.training_task.gradient_clip_norm is None
            else float(request.training_task.gradient_clip_norm)
        ),
        log_every_steps=int(trainer_options.log_every_steps),
        algorithm=runtime.algorithm,
        initial_query_ssl_algorithm_state=request.initial_query_ssl_algorithm_state,
        resume_checkpoint_path=trainer_options.resume_checkpoint_path,
        resume_checkpoint_output_dir=trainer_options.resume_checkpoint_output_dir,
        resume_checkpoint_every_epochs=int(
            trainer_options.resume_checkpoint_every_epochs
        ),
        before_restore_best=before_restore_best,
    )

    diagnostic_threshold = resolve_fixed_pseudo_label_diagnostic_threshold(
        request.query_ssl_config.parameters
    )
    pseudo_label_quality = build_final_snapshot_pseudo_label_quality(
        model=model,
        tokenizer=runtime.tokenizer,
        rows=(
            list(runtime.effective_unlabeled_rows)
            if request.diagnostic_unlabeled_rows is None
            else list(request.diagnostic_unlabeled_rows)
        ),
        labels=runtime.effective_labels,
        backbone_config=runtime.backbone_config,
        acceptance_threshold=diagnostic_threshold.threshold,
        trainer_runtime_config=request.trainer_runtime_config,
        unlabeled_batch_size=request.query_ssl_config.unlabeled_batch_size or 1,
        tokenization_cache=None,
        tokenization_cache_namespace=runtime.tokenization_cache_namespace,
    )

    return QuerySslFullTextEncoderLocalSslResult(
        model=model,
        tokenizer=runtime.tokenizer,
        algorithm=runtime.algorithm,
        effective_labeled_rows=runtime.effective_labeled_rows,
        effective_unlabeled_rows=runtime.effective_unlabeled_rows,
        effective_labels=runtime.effective_labels,
        selection_rows=runtime.selection_rows,
        local_step_plan=runtime.step_plan,
        history=tuple(history),
        best_selection_report=dict(best_selection_report),
        final_model_state_dict=final_model_state_dict,
        pseudo_label_quality=pseudo_label_quality,
        diagnostic_client_metrics=diagnostic_threshold.to_client_metrics(),
        tokenization_cache_namespace=runtime.tokenization_cache_namespace,
    )


def _clone_model_state_dict(model: TextEncoderWithLinearHead) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu().clone()
        else:
            state[key] = copy.deepcopy(value)
    return state


def _prepare_query_ssl_local_runtime(
    request: CentralQuerySslTextEncoderSessionRequest,
) -> _QuerySslFullTextEncoderLocalRuntime:
    effective_labeled_rows = tuple(request.labeled_rows)
    effective_unlabeled_rows = tuple(request.unlabeled_rows)
    if not effective_labeled_rows:
        raise ValueError("Query SSL full text encoder training requires labeled_rows.")
    if not effective_unlabeled_rows:
        raise ValueError(
            "Query SSL full text encoder training requires unlabeled_rows."
        )

    descriptor = resolve_query_ssl_algorithm_descriptor(
        request.query_ssl_config.algorithm_name
    )
    validate_query_ssl_unlabeled_views(
        rows=effective_unlabeled_rows,
        view_builder_name=descriptor.required_views.view_builder_name,
        algorithm_name=descriptor.algorithm_name,
    )
    algorithm = descriptor.build_algorithm(request.query_ssl_config.parameters)
    effective_labels = tuple(str(label) for label in request.labels)
    if not effective_labels:
        raise ValueError("Full text encoder/head label schema must not be empty.")
    _validate_labeled_rows_have_known_labels(
        rows=effective_labeled_rows,
        labels=effective_labels,
    )
    effective_selection_rows = (
        tuple(request.selection_rows)
        if request.selection_rows is not None
        else effective_labeled_rows
    )
    _validate_labeled_rows_have_known_labels(
        rows=effective_selection_rows,
        labels=effective_labels,
    )
    backbone_config = _build_backbone_config(request.cfg)
    cache_namespace = tokenization_cache_namespace(backbone_config)
    label_to_index = {label: index for index, label in enumerate(effective_labels)}
    train_loader = build_dataloader(
        rows=list(effective_labeled_rows),
        label_to_index=label_to_index,
        tokenizer=request.tokenizer,
        batch_size=int(request.training_task.batch_size),
        max_length=backbone_config.max_length,
        task_prefix=backbone_config.task_prefix,
        shuffle=True,
        tokenization_cache=None,
        tokenization_cache_namespace=cache_namespace,
        drop_last=getattr(request.query_ssl_config, "drop_last_train_batches", False),
    )
    selection_loader = build_dataloader(
        rows=list(effective_selection_rows),
        label_to_index=label_to_index,
        tokenizer=request.tokenizer,
        batch_size=int(request.training_task.batch_size),
        max_length=backbone_config.max_length,
        task_prefix=backbone_config.task_prefix,
        shuffle=False,
        tokenization_cache=None,
        tokenization_cache_namespace=cache_namespace,
        drop_last=getattr(
            request.query_ssl_config,
            "drop_last_unlabeled_batches",
            False,
        ),
    )
    unlabeled_loader = build_query_ssl_unlabeled_dataloader(
        rows=effective_unlabeled_rows,
        tokenizer=request.tokenizer,
        batch_size=request.query_ssl_config.unlabeled_batch_size
        or int(request.training_task.batch_size),
        max_length=backbone_config.max_length,
        task_prefix=backbone_config.task_prefix,
        shuffle=True,
        view_builder_name=descriptor.required_views.view_builder_name,
        strong_view_policy=request.query_ssl_config.strong_view_policy,
        tokenization_cache=None,
        tokenization_cache_namespace=cache_namespace,
        drop_last=getattr(
            request.query_ssl_config,
            "drop_last_unlabeled_batches",
            False,
        ),
    )
    step_plan = build_query_ssl_local_step_plan(
        labeled_loader_steps=len(train_loader),
        unlabeled_loader_steps=len(unlabeled_loader),
        uses_labeled_batches=algorithm.uses_labeled_batches,
        local_epochs=int(request.training_task.local_epochs),
        max_steps=int(request.training_task.max_steps),
    )
    return _QuerySslFullTextEncoderLocalRuntime(
        model=request.model,
        tokenizer=request.tokenizer,
        algorithm=algorithm,
        effective_labeled_rows=effective_labeled_rows,
        effective_unlabeled_rows=effective_unlabeled_rows,
        effective_labels=effective_labels,
        selection_rows=effective_selection_rows,
        train_loader=train_loader,
        selection_loader=selection_loader,
        unlabeled_loader=unlabeled_loader,
        step_plan=step_plan,
        backbone_config=backbone_config,
        tokenization_cache_namespace=cache_namespace,
    )


def _build_backbone_config(cfg: Any) -> SimpleNamespace:
    paper_backbone = getattr(cfg, "paper_backbone", None)
    return SimpleNamespace(
        max_length=int(getattr(paper_backbone, "max_length", 256)),
        task_prefix=str(getattr(paper_backbone, "task_prefix", "") or ""),
        tokenizer_model_id=str(
            getattr(
                paper_backbone,
                "tokenizer_model_id",
                getattr(paper_backbone, "model_id", ""),
            )
        ),
        tokenizer_revision=str(getattr(paper_backbone, "tokenizer_revision", "main")),
    )


def _validate_labeled_rows_have_known_labels(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
) -> None:
    known_labels = {str(label) for label in labels}
    missing = sorted({str(row["mapped_label_4"]) for row in rows} - known_labels)
    if missing:
        raise ValueError(
            "Query SSL labeled_rows contain labels outside active label_schema: "
            f"{missing}."
        )
