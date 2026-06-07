"""PEFT text encoder Query SSL local training session."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingTask

from .delta_extraction import load_peft_encoder_base_parameters_into_model
from .loops import set_seed, train_query_ssl_classifier
from .modeling import (
    PeftTextEncoderWithLinearHead,
    build_peft_text_encoder_with_linear_head_from_config,
)
from .pseudo_label_diagnostics import (
    build_final_snapshot_pseudo_label_quality,
    resolve_fixed_pseudo_label_diagnostic_threshold,
    tokenization_cache_namespace,
)


class QuerySslPeftEncoderObjectiveRuntimeConfig(Protocol):
    """Query SSL PEFT encoder local core가 필요한 objective config surface."""

    algorithm_name: str
    parameters: Mapping[str, object]
    strong_view_policy: str
    unlabeled_batch_size: int | None
    drop_last_train_batches: bool
    drop_last_unlabeled_batches: bool


class PeftEncoderTrainerRuntimeConfig(Protocol):
    """PEFT text encoder/head 모델 로딩/학습 core가 필요한 runtime config surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


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
    pseudo_label_quality: PseudoLabelQualitySummary
    diagnostic_client_metrics: Mapping[str, float]
    tokenization_cache: TextTokenizationCache | None
    tokenization_cache_namespace: str


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderLocalTrainerOptions:
    """문맥별 runner가 공통 local session에 넘기는 trainer 실행 옵션."""

    classifier_learning_rate: float | None = None
    weight_decay: float = 0.0
    log_every_steps: int = 0
    resume_checkpoint_path: str | Path | None = None
    resume_checkpoint_output_dir: str | Path | None = None
    resume_checkpoint_every_epochs: int = 0


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
        pseudo_label_quality=pseudo_label_quality,
        diagnostic_client_metrics=diagnostic_threshold.to_client_metrics(),
        tokenization_cache=runtime.tokenization_cache,
        tokenization_cache_namespace=runtime.tokenization_cache_namespace,
    )


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
