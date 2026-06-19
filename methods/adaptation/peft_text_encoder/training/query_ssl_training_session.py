"""PEFT text encoder Query SSL local training session."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.query_text_views.data import (
    build_dataloader,
)
from methods.adaptation.query_text_views.local_training_budget import (
    QuerySslLocalStepPlan,
    build_query_ssl_local_step_plan,
)
from methods.adaptation.query_text_views.query_ssl_views import (
    build_query_ssl_unlabeled_dataloader,
)
from methods.adaptation.query_text_views.tokenization import (
    TextTokenizationCache,
    resolve_text_tokenization_cache,
)
from methods.adaptation.query_text_views.view_rows import (
    validate_query_ssl_unlabeled_views,
)
from methods.adaptation.text_encoder_classifier.query_ssl_session import (
    CentralQuerySslTextEncoderSessionRequest,
)
from methods.adaptation.text_encoder_classifier.query_ssl_training import (
    set_seed,
    train_query_ssl_classifier,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingTask

from .delta_extraction import (
    extract_peft_encoder_materialized_state,
    load_peft_encoder_base_parameters_into_model,
)
from .local_training_surface import (
    PeftEncoderTrainerRuntimeConfig,
    QuerySslPeftEncoderLocalSessionRequest,
    QuerySslPeftEncoderLocalTrainerOptions,
    QuerySslPeftEncoderObjectiveRuntimeConfig,
)
from .modeling import (
    PeftTextEncoderWithLinearHead,
    build_peft_text_encoder_with_linear_head_from_config,
)
from .pseudo_label_diagnostics import (
    build_final_snapshot_pseudo_label_quality,
    resolve_fixed_pseudo_label_diagnostic_threshold,
    tokenization_cache_namespace,
)


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderLocalSslResult:
    """중앙/FL이 공유할 수 있는 PEFT Query SSL local 학습 결과."""

    model: PeftTextEncoderWithLinearHead
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
    tokenization_cache: TextTokenizationCache | None
    tokenization_cache_namespace: str


@dataclass(frozen=True, slots=True)
class _QuerySslPeftEncoderLocalRuntime:
    """PEFT Query SSL local 학습 loop에 필요한 준비된 runtime."""

    model: PeftTextEncoderWithLinearHead
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
    tokenization_cache: TextTokenizationCache | None
    tokenization_cache_namespace: str


def run_query_ssl_peft_encoder_local_ssl(
    *,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    selection_rows: Sequence[LabeledQueryRow] | None = None,
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    training_task: TrainingTask,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig,
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
    trainer_options: QuerySslPeftEncoderLocalTrainerOptions | None = None,
) -> QuerySslPeftEncoderLocalSslResult:
    """PEFT text encoder Query SSL local 학습을 실행한다."""

    effective_trainer_options = (
        QuerySslPeftEncoderLocalTrainerOptions()
        if trainer_options is None
        else trainer_options
    )
    runtime = _prepare_query_ssl_local_runtime(
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        selection_rows=selection_rows,
        labels=labels,
        base_parameters=base_parameters,
        training_task=training_task,
        query_ssl_config=query_ssl_config,
        peft_config=peft_config,
        trainer_runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
        timing_recorder=timing_recorder,
    )

    final_model_state_dict: dict[str, Any] | None = None

    def capture_final_model_state(model: PeftTextEncoderWithLinearHead) -> None:
        nonlocal final_model_state_dict
        final_model_state_dict = _clone_model_state_dict(model)

    before_restore_best = (
        capture_final_model_state
        if effective_trainer_options.capture_final_model_state
        else None
    )

    with _measure(timing_recorder, "core_training_loop_seconds"):
        model, history, _best_selection_report = train_query_ssl_classifier(
            model=runtime.model,
            train_loader=runtime.train_loader,
            unlabeled_loader=runtime.unlabeled_loader,
            selection_loader=runtime.selection_loader,
            categories=list(runtime.effective_labels),
            device=trainer_runtime_config.device,
            epochs=int(training_task.local_epochs),
            max_train_steps=runtime.step_plan.total_steps,
            learning_rate=float(training_task.learning_rate),
            classifier_learning_rate=(
                float(training_task.learning_rate)
                if effective_trainer_options.classifier_learning_rate is None
                else float(effective_trainer_options.classifier_learning_rate)
            ),
            weight_decay=float(effective_trainer_options.weight_decay),
            proximal_mu=float(peft_config.proximal_mu),
            max_grad_norm=_training_gradient_clip_norm(training_task),
            log_every_steps=int(effective_trainer_options.log_every_steps),
            algorithm=runtime.algorithm,
            initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
            resume_checkpoint_path=effective_trainer_options.resume_checkpoint_path,
            resume_checkpoint_output_dir=(
                effective_trainer_options.resume_checkpoint_output_dir
            ),
            resume_checkpoint_every_epochs=int(
                effective_trainer_options.resume_checkpoint_every_epochs
            ),
            before_restore_best=before_restore_best,
        )

    diagnostic_threshold = resolve_fixed_pseudo_label_diagnostic_threshold(
        query_ssl_config.parameters
    )
    with _measure(timing_recorder, "core_pseudo_label_diagnostics_seconds"):
        pseudo_label_quality = build_final_snapshot_pseudo_label_quality(
            model=model,
            tokenizer=runtime.tokenizer,
            rows=(
                list(runtime.effective_unlabeled_rows)
                if diagnostic_unlabeled_rows is None
                else list(diagnostic_unlabeled_rows)
            ),
            labels=runtime.effective_labels,
            peft_config=peft_config,
            acceptance_threshold=diagnostic_threshold.threshold,
            trainer_runtime_config=trainer_runtime_config,
            unlabeled_batch_size=query_ssl_config.unlabeled_batch_size or 1,
            tokenization_cache=runtime.tokenization_cache,
            tokenization_cache_namespace=runtime.tokenization_cache_namespace,
        )

    return QuerySslPeftEncoderLocalSslResult(
        model=model,
        tokenizer=runtime.tokenizer,
        algorithm=runtime.algorithm,
        effective_labeled_rows=runtime.effective_labeled_rows,
        effective_unlabeled_rows=runtime.effective_unlabeled_rows,
        effective_labels=runtime.effective_labels,
        selection_rows=runtime.selection_rows,
        local_step_plan=runtime.step_plan,
        history=tuple(history),
        best_selection_report=dict(_best_selection_report),
        final_model_state_dict=final_model_state_dict,
        pseudo_label_quality=pseudo_label_quality,
        diagnostic_client_metrics=diagnostic_threshold.to_client_metrics(),
        tokenization_cache=runtime.tokenization_cache,
        tokenization_cache_namespace=runtime.tokenization_cache_namespace,
    )


def run_query_ssl_peft_encoder_local_session(
    request: QuerySslPeftEncoderLocalSessionRequest,
) -> QuerySslPeftEncoderLocalSslResult:
    """공통 request surface로 PEFT text encoder Query SSL local 학습을 실행한다."""

    return run_query_ssl_peft_encoder_local_ssl(
        seed=request.seed,
        labeled_rows=request.labeled_rows,
        unlabeled_rows=request.unlabeled_rows,
        diagnostic_unlabeled_rows=request.diagnostic_unlabeled_rows,
        selection_rows=request.selection_rows,
        labels=request.labels,
        base_parameters=request.base_parameters,
        training_task=request.training_task,
        query_ssl_config=request.query_ssl_config,
        peft_config=request.peft_config,
        trainer_runtime_config=request.trainer_runtime_config,
        runtime_resource_cache=request.runtime_resource_cache,
        timing_recorder=request.timing_recorder,
        initial_query_ssl_algorithm_state=request.initial_query_ssl_algorithm_state,
        trainer_options=request.trainer_options,
    )


def run_central_query_ssl_peft_encoder_session(
    request: CentralQuerySslTextEncoderSessionRequest,
) -> QuerySslPeftEncoderLocalSslResult:
    """중앙 surface-neutral 요청을 PEFT local training core 입력으로 변환한다."""

    labels = tuple(str(label) for label in request.labels)
    return run_query_ssl_peft_encoder_local_ssl(
        seed=request.seed,
        labeled_rows=request.labeled_rows,
        unlabeled_rows=request.unlabeled_rows,
        diagnostic_unlabeled_rows=request.diagnostic_unlabeled_rows,
        selection_rows=request.selection_rows,
        labels=labels,
        base_parameters=extract_peft_encoder_materialized_state(
            model=request.model,
            labels=labels,
        ),
        training_task=request.training_task,
        query_ssl_config=request.query_ssl_config,
        peft_config=_build_central_peft_config(
            cfg=request.cfg,
            labels=labels,
        ),
        trainer_runtime_config=request.trainer_runtime_config,
        initial_query_ssl_algorithm_state=request.initial_query_ssl_algorithm_state,
        trainer_options=QuerySslPeftEncoderLocalTrainerOptions(
            classifier_learning_rate=(
                None
                if request.trainer_options is None
                else request.trainer_options.classifier_learning_rate
            ),
            weight_decay=(
                0.0
                if request.trainer_options is None
                else request.trainer_options.weight_decay
            ),
            log_every_steps=(
                0
                if request.trainer_options is None
                else request.trainer_options.log_every_steps
            ),
            resume_checkpoint_path=(
                None
                if request.trainer_options is None
                else request.trainer_options.resume_checkpoint_path
            ),
            resume_checkpoint_output_dir=(
                None
                if request.trainer_options is None
                else request.trainer_options.resume_checkpoint_output_dir
            ),
            resume_checkpoint_every_epochs=(
                0
                if request.trainer_options is None
                else request.trainer_options.resume_checkpoint_every_epochs
            ),
            capture_final_model_state=(
                False
                if request.trainer_options is None
                else request.trainer_options.capture_final_model_state
            ),
        ),
    )


def _clone_model_state_dict(model: PeftTextEncoderWithLinearHead) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu().clone()
        else:
            state[key] = copy.deepcopy(value)
    return state


def _build_central_peft_config(
    *,
    cfg: Any,
    labels: Sequence[str],
) -> PeftEncoderTrainingBackendConfig:
    defaults = PeftEncoderTrainingBackendConfig()
    paper_backbone = getattr(cfg, "paper_backbone", None)
    peft_adapter = getattr(cfg, "peft_adapter", None)
    return PeftEncoderTrainingBackendConfig(
        backbone_model_id=_optional_str_attr(
            paper_backbone,
            "model_id",
            defaults.backbone_model_id,
        ),
        backbone_revision=_optional_str_attr(
            paper_backbone,
            "revision",
            defaults.backbone_revision,
        ),
        tokenizer_model_id=_optional_str_attr(
            paper_backbone,
            "tokenizer_model_id",
            defaults.tokenizer_model_id,
        ),
        tokenizer_revision=_optional_str_attr(
            paper_backbone,
            "tokenizer_revision",
            defaults.tokenizer_revision,
        ),
        pooling=_optional_str_attr(paper_backbone, "pooling", defaults.pooling),
        max_length=int(getattr(paper_backbone, "max_length", defaults.max_length)),
        task_prefix=_optional_str_attr(
            paper_backbone,
            "task_prefix",
            defaults.task_prefix,
        ),
        peft_adapter_name=_optional_str_attr(
            peft_adapter,
            "peft_adapter_name",
            defaults.peft_adapter_name,
        ),
        rank=int(getattr(peft_adapter, "rank", defaults.rank)),
        alpha=int(getattr(peft_adapter, "alpha", defaults.alpha)),
        dropout=float(getattr(peft_adapter, "dropout", defaults.dropout)),
        bias=_optional_str_attr(peft_adapter, "bias", defaults.bias),
        target_modules=_optional_str_attr(
            peft_adapter,
            "target_modules",
            defaults.target_modules,
        ),
        use_rslora=bool(getattr(peft_adapter, "use_rslora", defaults.use_rslora)),
        delta_format=defaults.delta_format,
        artifact_ref_prefix=defaults.artifact_ref_prefix,
        label_schema=tuple(str(label) for label in labels),
    )


def _optional_str_attr(source: Any, key: str, default: str) -> str:
    value = getattr(source, key, None)
    if value is None:
        return default
    normalized = str(value).strip()
    return normalized if normalized else default


def _prepare_query_ssl_local_runtime(
    *,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    selection_rows: Sequence[LabeledQueryRow] | None,
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    training_task: TrainingTask,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig,
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
    timing_recorder: TimingRecorder | None,
) -> _QuerySslPeftEncoderLocalRuntime:
    """row/model/dataloader를 Query SSL local 학습 loop 입력으로 정규화한다."""

    effective_labeled_rows = tuple(labeled_rows)
    effective_unlabeled_rows = tuple(unlabeled_rows)
    if not effective_labeled_rows:
        raise ValueError("Query SSL PEFT encoder local training requires labeled_rows.")
    if not effective_unlabeled_rows:
        raise ValueError(
            "Query SSL PEFT encoder local training requires unlabeled_rows."
        )

    descriptor = resolve_query_ssl_algorithm_descriptor(query_ssl_config.algorithm_name)
    validate_query_ssl_unlabeled_views(
        rows=effective_unlabeled_rows,
        view_builder_name=descriptor.required_views.view_builder_name,
        algorithm_name=descriptor.algorithm_name,
    )
    algorithm = descriptor.build_algorithm(query_ssl_config.parameters)
    effective_labels = tuple(str(label) for label in labels)
    if not effective_labels:
        raise ValueError("PEFT text encoder/head label schema must not be empty.")
    _validate_labeled_rows_have_known_labels(
        rows=effective_labeled_rows,
        labels=effective_labels,
    )
    effective_selection_rows = (
        tuple(selection_rows) if selection_rows is not None else None
    )
    if effective_selection_rows is not None:
        _validate_labeled_rows_have_known_labels(
            rows=effective_selection_rows,
            labels=effective_labels,
        )
    tokenization_cache = resolve_text_tokenization_cache(runtime_resource_cache)
    tokenization_cache_namespace_value = tokenization_cache_namespace(peft_config)

    with _measure(timing_recorder, "core_model_prepare_seconds"):
        with _measure(timing_recorder, "core_seed_seconds"):
            set_seed(int(seed))
        with _measure(timing_recorder, "core_model_build_seconds"):
            model, tokenizer = _build_peft_encoder_model(
                labels=effective_labels,
                peft_config=peft_config,
                trainer_runtime_config=trainer_runtime_config,
                runtime_resource_cache=runtime_resource_cache,
            )
        with _measure(timing_recorder, "core_base_parameter_load_seconds"):
            load_peft_encoder_base_parameters_into_model(
                model=model,
                labels=effective_labels,
                base_parameters=base_parameters,
                device=trainer_runtime_config.device,
            )

    with _measure(timing_recorder, "core_dataloader_prepare_seconds"):
        label_to_index = {label: index for index, label in enumerate(effective_labels)}
        prepared_selection_rows = (
            effective_selection_rows
            if effective_selection_rows is not None
            else tuple(
                _build_bounded_label_balanced_selection_rows(
                    rows=effective_labeled_rows,
                    max_examples=training_task.selection_policy.max_examples,
                    seed=int(seed),
                )
            )
        )
        train_loader = build_dataloader(
            rows=effective_labeled_rows,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=int(training_task.batch_size),
            max_length=peft_config.max_length,
            task_prefix=peft_config.task_prefix,
            shuffle=True,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
            drop_last=getattr(
                query_ssl_config,
                "drop_last_train_batches",
                False,
            ),
        )
        selection_loader = build_dataloader(
            rows=prepared_selection_rows,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            batch_size=int(training_task.batch_size),
            max_length=peft_config.max_length,
            task_prefix=peft_config.task_prefix,
            shuffle=False,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
            drop_last=getattr(
                query_ssl_config,
                "drop_last_unlabeled_batches",
                False,
            ),
        )
        unlabeled_loader = _build_unlabeled_loader(
            rows=effective_unlabeled_rows,
            tokenizer=tokenizer,
            batch_size=query_ssl_config.unlabeled_batch_size
            or int(training_task.batch_size),
            max_length=peft_config.max_length,
            task_prefix=peft_config.task_prefix,
            strong_view_policy=query_ssl_config.strong_view_policy,
            view_builder_name=descriptor.required_views.view_builder_name,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
        )
        step_plan = build_query_ssl_local_step_plan(
            labeled_loader_steps=len(train_loader),
            unlabeled_loader_steps=len(unlabeled_loader),
            uses_labeled_batches=algorithm.uses_labeled_batches,
            local_epochs=int(training_task.local_epochs),
            max_steps=int(training_task.max_steps),
        )

    return _QuerySslPeftEncoderLocalRuntime(
        model=model,
        tokenizer=tokenizer,
        algorithm=algorithm,
        effective_labeled_rows=effective_labeled_rows,
        effective_unlabeled_rows=effective_unlabeled_rows,
        effective_labels=effective_labels,
        selection_rows=prepared_selection_rows,
        train_loader=train_loader,
        selection_loader=selection_loader,
        unlabeled_loader=unlabeled_loader,
        step_plan=step_plan,
        tokenization_cache=tokenization_cache,
        tokenization_cache_namespace=tokenization_cache_namespace_value,
    )


def _build_peft_encoder_model(
    *,
    labels: Sequence[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
) -> tuple[PeftTextEncoderWithLinearHead, Any]:
    return build_peft_text_encoder_with_linear_head_from_config(
        labels=[str(label) for label in labels],
        peft_config=peft_config,
        runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )


def _measure(timing_recorder: TimingRecorder | None, key: str) -> Any:
    if timing_recorder is None:
        return nullcontext()
    return timing_recorder.measure(key)


def _build_bounded_label_balanced_selection_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    max_examples: int | None,
    seed: int,
) -> list[LabeledQueryRow]:
    """client-local selection 평가용 labeled probe를 class-balanced로 제한한다."""

    source_rows = list(rows)
    if max_examples is None or max_examples <= 0 or max_examples >= len(source_rows):
        return source_rows

    rng = random.Random(seed)
    rows_by_label: dict[str, list[LabeledQueryRow]] = {}
    for row in source_rows:
        rows_by_label.setdefault(str(row["mapped_label_4"]), []).append(row)
    for label_rows in rows_by_label.values():
        rng.shuffle(label_rows)

    selected: list[LabeledQueryRow] = []
    label_order = sorted(rows_by_label)
    while len(selected) < max_examples and label_order:
        next_label_order: list[str] = []
        for label in label_order:
            bucket = rows_by_label[label]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if len(selected) >= max_examples:
                break
            if bucket:
                next_label_order.append(label)
        label_order = next_label_order
    return selected


def _training_gradient_clip_norm(training_task: TrainingTask) -> float:
    if training_task.gradient_clip_norm is None:
        return 0.0
    return float(training_task.gradient_clip_norm)


def _build_unlabeled_loader(
    *,
    rows: Sequence[LabeledQueryRow],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    task_prefix: str,
    strong_view_policy: str,
    view_builder_name: str,
    tokenization_cache: TextTokenizationCache | None,
    tokenization_cache_namespace: str,
    drop_last: bool = False,
) -> Any:
    return build_query_ssl_unlabeled_dataloader(
        rows=rows,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        task_prefix=task_prefix,
        shuffle=True,
        view_builder_name=view_builder_name,
        strong_view_policy=strong_view_policy,
        tokenization_cache=tokenization_cache,
        tokenization_cache_namespace=tokenization_cache_namespace,
        drop_last=drop_last,
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
