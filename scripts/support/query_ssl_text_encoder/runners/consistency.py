"""Query SSL consistency family runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from methods.adaptation.peft_text_encoder.config import (
    PEFT_ENCODER_TRAINING_BACKEND_NAME,
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    extract_peft_encoder_materialized_state,
)
from methods.adaptation.peft_text_encoder.training.query_ssl_training_session import (
    QuerySslPeftEncoderLocalSslResult,
    QuerySslPeftEncoderLocalTrainerOptions,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.query_text_views.data import DEFAULT_STRONG_VIEW_POLICY
from methods.adaptation.query_text_views.unlabeled_preparation import (
    PreparedQuerySslUnlabeledRows,
)
from methods.ssl.base import (
    QUERY_SSL_INPUT_TRANSFORM_NONE,
    QUERY_SSL_MODEL_OUTPUT_LOGITS,
    QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_POST_STEP_HOOK,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QUERY_SSL_TEACHER_STATE_EMA_TRAINABLE,
    QUERY_SSL_TEACHER_STATE_NONE,
    QuerySslAlgorithm,
    QuerySslAlgorithmDescriptor,
)
from methods.ssl.model_capabilities import require_pooled_feature_classifier
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from methods.ssl.state import export_query_ssl_algorithm_report_state_summary
from scripts.support.configured_callable import load_configured_callable
from scripts.support.query_ssl_text_encoder.io.artifacts import write_run_artifacts
from scripts.support.query_ssl_text_encoder.query_ssl.run_context import (
    QuerySslRunContext,
    build_query_ssl_method_manifest,
    build_query_ssl_method_parameters,
    evaluate_query_ssl_run_context,
    prepare_query_ssl_run_context,
)
from scripts.support.query_ssl_text_encoder.query_ssl.view_preparation import (
    build_query_ssl_augmenter_manifest,
    prepare_query_ssl_unlabeled_rows,
)
from scripts.support.query_ssl_text_encoder.result_utils import (
    extract_final_selection_report,
    merge_results_with_best_and_final,
)
from scripts.support.query_ssl_text_encoder.runtime_metrics import (
    run_with_training_runtime_metrics,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)


@dataclass(frozen=True, slots=True)
class _CentralSslSurfaceRuntime:
    """중앙 SSL runner가 trainable_surface leaf에서 읽은 runtime callable."""

    surface_name: str
    model_builder: Callable[..., tuple[Any, Any, dict[str, Any]]]
    local_session_runner: Callable[..., QuerySslPeftEncoderLocalSslResult]
    trainer_version_prefix: str


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

    (
        prepared_unlabeled_rows,
        context,
        algorithm,
        surface_runtime,
    ) = _prepare_query_ssl_training_runtime(
        cfg=cfg,
        descriptor=descriptor,
        train_rows=train_rows,
        unlabeled_rows=unlabeled_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
    )
    max_train_steps = _resolve_max_train_steps(cfg)
    local_ssl_result, runtime_metrics = _train_query_ssl_context(
        cfg=cfg,
        context=context,
        algorithm=algorithm,
        surface_runtime=surface_runtime,
        max_train_steps=max_train_steps,
    )
    results = evaluate_query_ssl_run_context(
        model=local_ssl_result.model,
        eval_loaders=context.eval_loaders,
        categories=context.categories,
        device=context.training_device,
    )
    history = list(local_ssl_result.history)
    final_selection_report = extract_final_selection_report(history=history)
    final_results = merge_results_with_best_and_final(
        results=results,
        selection_set=context.effective_selection_set,
        final_selection_report=final_selection_report,
    )
    effective_extra_manifest = _build_query_ssl_extra_manifest(
        cfg=cfg,
        context=context,
        prepared_unlabeled_rows=prepared_unlabeled_rows,
        algorithm=local_ssl_result.algorithm,
        runtime_metrics=runtime_metrics,
        extra_manifest=extra_manifest,
    )
    outputs = write_run_artifacts(
        cfg=context.cfg,
        trainer_version=context.trainer_version,
        created_at=context.created_at,
        model=local_ssl_result.model,
        tokenizer=local_ssl_result.tokenizer,
        categories=context.categories,
        eval_set_map=context.eval_set_map,
        training_device=context.training_device,
        backbone_summary=context.backbone_summary,
        history=history,
        best_selection_report=dict(local_ssl_result.best_selection_report),
        final_selection_report=(
            dict(final_selection_report) if final_selection_report is not None else None
        ),
        results=final_results,
        extra_manifest=effective_extra_manifest,
        eval_loaders=context.eval_loaders,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return outputs


def _prepare_query_ssl_training_runtime(
    *,
    cfg: Any,
    descriptor: QuerySslAlgorithmDescriptor,
    train_rows: list[LabeledQueryRow] | None,
    unlabeled_rows: list[LabeledQueryRow] | None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    selection_set_name: str | None,
    categories_override: list[str] | tuple[str, ...] | None,
) -> tuple[
    PreparedQuerySslUnlabeledRows,
    QuerySslRunContext,
    QuerySslAlgorithm,
    _CentralSslSurfaceRuntime,
]:
    """unlabeled view, run context, loader, algorithm runtime을 준비한다."""

    surface_runtime = _resolve_trainable_surface_runtime(cfg)
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
        model_builder=surface_runtime.model_builder,
        trainer_version_prefix=_build_trainer_version_prefix(
            descriptor=descriptor,
            surface_runtime=surface_runtime,
        ),
        algorithm_name=descriptor.display_name,
    )
    _validate_query_ssl_runner_capabilities(
        descriptor=descriptor,
        model=context.model,
        surface_runtime=surface_runtime,
    )
    algorithm = descriptor.build_algorithm(build_query_ssl_method_parameters(cfg))
    return (
        prepared_unlabeled_rows,
        context,
        algorithm,
        surface_runtime,
    )


def _train_query_ssl_context(
    *,
    cfg: Any,
    context: QuerySslRunContext,
    algorithm: QuerySslAlgorithm,
    surface_runtime: _CentralSslSurfaceRuntime,
    max_train_steps: int | None,
) -> tuple[QuerySslPeftEncoderLocalSslResult, dict[str, Any]]:
    """준비된 Query SSL context로 학습을 실행하고 runtime metric을 반환한다."""

    local_session_request = _build_central_local_session_request(
        cfg=cfg,
        context=context,
        max_train_steps=max_train_steps,
    )
    local_ssl_result, runtime_metrics = run_with_training_runtime_metrics(
        lambda: surface_runtime.local_session_runner(
            **local_session_request,
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
    return local_ssl_result, runtime_metrics


def _build_central_local_session_request(
    *,
    cfg: Any,
    context: QuerySslRunContext,
    max_train_steps: int | None,
) -> dict[str, Any]:
    """중앙 pooled SSL cfg/context를 공통 local session 입력으로 변환한다."""

    return {
        "seed": int(cfg.seed),
        "labeled_rows": list(context.effective_train_rows),
        "unlabeled_rows": list(context.effective_unlabeled_rows),
        "diagnostic_unlabeled_rows": list(context.effective_unlabeled_rows),
        "selection_rows": _load_selection_rows(context),
        "labels": list(context.categories),
        "base_parameters": _extract_central_base_parameters(context),
        "training_task": _build_central_training_task(
            cfg=cfg,
            context=context,
            max_train_steps=max_train_steps,
        ),
        "query_ssl_config": _build_central_query_ssl_config(cfg),
        "peft_config": _build_central_peft_config(
            cfg=context.cfg,
            labels=context.categories,
        ),
        "trainer_runtime_config": _build_central_trainer_runtime_config(
            cfg=context.cfg,
            device=context.training_device,
        ),
        "trainer_options": QuerySslPeftEncoderLocalTrainerOptions(
            classifier_learning_rate=float(cfg.classifier_learning_rate),
            weight_decay=float(cfg.weight_decay),
            log_every_steps=int(cfg.log_every_steps),
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
    }


def _extract_central_base_parameters(
    context: QuerySslRunContext,
) -> PeftEncoderMaterializedState:
    return extract_peft_encoder_materialized_state(
        model=context.model,
        labels=context.categories,
    )


def _load_selection_rows(context: QuerySslRunContext) -> list[LabeledQueryRow]:
    selection_name = context.effective_selection_set
    if context.eval_rows_by_name is not None:
        return list(context.eval_rows_by_name[selection_name])
    selection_path = context.eval_set_map[selection_name]
    return load_labeled_query_rows(selection_path)


def _build_central_training_task(
    *,
    cfg: Any,
    context: QuerySslRunContext,
    max_train_steps: int | None,
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        round_id="central_ssl_control",
        task_id=f"central_ssl_{context.trainer_version}",
        model_id=str(context.backbone_summary["backbone_model_id"]),
        model_revision=str(context.backbone_summary["backbone_revision"]),
        task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
        training_scope="adapter_only",
        local_epochs=int(cfg.epochs),
        batch_size=int(cfg.train_batch_size),
        learning_rate=float(cfg.learning_rate),
        max_steps=int(max_train_steps or int(cfg.epochs)),
        objective_config=TrainingObjectiveConfig.from_mapping(
            {"training_backend_name": PEFT_ENCODER_TRAINING_BACKEND_NAME}
        ),
        selection_policy=TrainingSelectionPolicy.from_mapping(
            {"max_examples": len(context.effective_train_rows)}
        ),
        gradient_clip_norm=float(cfg.max_grad_norm),
    )


def _build_central_query_ssl_config(cfg: Any) -> SimpleNamespace:
    return SimpleNamespace(
        algorithm_name=str(cfg.query_ssl_method.algorithm_name),
        parameters=build_query_ssl_method_parameters(cfg),
        strong_view_policy=_resolve_strong_view_policy(cfg),
        unlabeled_batch_size=int(cfg.query_ssl_method.unlabeled_batch_size),
        drop_last_train_batches=bool(
            getattr(cfg, "drop_last_train_batches", False)
        ),
        drop_last_unlabeled_batches=bool(
            getattr(cfg, "drop_last_unlabeled_batches", False)
        ),
    )


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


def _build_central_trainer_runtime_config(
    *,
    cfg: Any,
    device: str,
) -> SimpleNamespace:
    paper_backbone = getattr(cfg, "paper_backbone", None)
    runtime = getattr(cfg, "runtime", None)
    return SimpleNamespace(
        device=device,
        classifier_dropout=float(
            getattr(
                paper_backbone,
                "classifier_dropout",
                0.1,
            )
        ),
        cache_dir=getattr(paper_backbone, "cache_dir", None),
        local_files_only=bool(getattr(runtime, "local_files_only", False)),
        trust_remote_code=bool(getattr(paper_backbone, "trust_remote_code", False)),
    )


def _optional_str_attr(source: Any, key: str, default: str) -> str:
    value = getattr(source, key, None)
    if value is None:
        return default
    normalized = str(value).strip()
    return normalized if normalized else default


def _build_query_ssl_extra_manifest(
    *,
    cfg: Any,
    context: QuerySslRunContext,
    prepared_unlabeled_rows: PreparedQuerySslUnlabeledRows,
    algorithm: QuerySslAlgorithm,
    runtime_metrics: Mapping[str, Any],
    extra_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Query SSL run artifact manifest의 추가 metadata를 조립한다."""

    effective_extra_manifest: dict[str, Any] = {
        "unlabeled_jsonl": None
        if getattr(cfg, "unlabeled_jsonl", None) is None
        else str(cfg.unlabeled_jsonl),
        "unlabeled_row_count": len(context.effective_unlabeled_rows),
        "drop_last_unlabeled_batches": bool(
            getattr(cfg, "drop_last_unlabeled_batches", False)
        ),
        "query_ssl_method": build_query_ssl_method_manifest(cfg),
        "query_ssl_algorithm_state_summary": dict(
            export_query_ssl_algorithm_report_state_summary(algorithm)
        ),
        "query_ssl_resume": _build_query_ssl_resume_manifest(cfg),
        "runtime_metrics": runtime_metrics,
    }
    effective_extra_manifest.update(context.initial_checkpoint_manifest)
    if (
        prepared_unlabeled_rows.uses_strong_view_candidates
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
    return effective_extra_manifest


def _resolve_max_train_steps(cfg: Any) -> int | None:
    raw_value = getattr(cfg, "max_train_steps", None)
    if raw_value is None:
        return None
    return int(raw_value)


def _validate_query_ssl_runner_capabilities(
    *,
    descriptor: QuerySslAlgorithmDescriptor,
    model: Any,
    surface_runtime: _CentralSslSurfaceRuntime,
) -> None:
    """현재 중앙 SSL surface runtime이 지원하는 descriptor capability를 검증한다."""

    requirements = descriptor.runtime_requirements
    supported_model_outputs = frozenset(
        {
            QUERY_SSL_MODEL_OUTPUT_LOGITS,
            QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
        }
    )
    unsupported_model_outputs = requirements.model_outputs - supported_model_outputs
    if unsupported_model_outputs:
        raise ValueError(
            "Unsupported Query SSL model outputs for central SSL surface "
            f"{surface_runtime.surface_name!r}: "
            f"{sorted(unsupported_model_outputs)}."
        )
    if QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES in requirements.model_outputs:
        require_pooled_feature_classifier(model)

    supported_optimizer_lifecycle = frozenset(
        {
            QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
            QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
            QUERY_SSL_OPTIMIZER_LIFECYCLE_POST_STEP_HOOK,
        }
    )
    unsupported_optimizer_lifecycle = (
        requirements.optimizer_lifecycle - supported_optimizer_lifecycle
    )
    if unsupported_optimizer_lifecycle:
        raise ValueError(
            "Unsupported Query SSL optimizer lifecycle for central SSL surface "
            f"{surface_runtime.surface_name!r}: "
            f"{sorted(unsupported_optimizer_lifecycle)}."
        )
    if requirements.input_transform_surface != QUERY_SSL_INPUT_TRANSFORM_NONE:
        raise ValueError(
            "Unsupported Query SSL input transform for central SSL surface "
            f"{surface_runtime.surface_name!r}: "
            f"{requirements.input_transform_surface!r}."
        )
    supported_teacher_states = frozenset(
        {
            QUERY_SSL_TEACHER_STATE_NONE,
            QUERY_SSL_TEACHER_STATE_EMA_TRAINABLE,
        }
    )
    if requirements.teacher_state not in supported_teacher_states:
        raise ValueError(
            "Unsupported Query SSL teacher state for central SSL surface "
            f"{surface_runtime.surface_name!r}: "
            f"{requirements.teacher_state!r}."
        )


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


def _resolve_strong_view_policy(cfg: Any) -> str:
    raw_policy = getattr(cfg, "query_ssl_strong_view_policy", None)
    if raw_policy is None:
        return DEFAULT_STRONG_VIEW_POLICY
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


def _build_trainer_version_prefix(
    *,
    descriptor: QuerySslAlgorithmDescriptor,
    surface_runtime: _CentralSslSurfaceRuntime,
) -> str:
    surface_prefix = surface_runtime.trainer_version_prefix.strip().lower()
    algorithm_name = descriptor.algorithm_name.strip().lower()
    return f"{surface_prefix}_{algorithm_name}"


def _resolve_trainable_surface_runtime(cfg: Any) -> _CentralSslSurfaceRuntime:
    trainable_surface = getattr(cfg, "trainable_surface", None)
    if trainable_surface is None:
        raise ValueError("trainable_surface config is required for central SSL.")
    surface_name = _read_required_config_str(
        trainable_surface,
        "name",
        field_name="trainable_surface.name",
    )
    central_ssl = getattr(trainable_surface, "central_ssl", None)
    if central_ssl is None:
        raise ValueError(
            "trainable_surface.central_ssl is required for central SSL: "
            f"surface={surface_name!r}."
        )
    return _CentralSslSurfaceRuntime(
        surface_name=surface_name,
        model_builder=load_configured_callable(
            _read_required_config_str(
                central_ssl,
                "model_builder",
                field_name="trainable_surface.central_ssl.model_builder",
            ),
            field_name="trainable_surface.central_ssl.model_builder",
        ),
        local_session_runner=load_configured_callable(
            _read_required_config_str(
                central_ssl,
                "local_session_runner",
                field_name="trainable_surface.central_ssl.local_session_runner",
            ),
            field_name="trainable_surface.central_ssl.local_session_runner",
        ),
        trainer_version_prefix=_read_required_config_str(
            central_ssl,
            "trainer_version_prefix",
            field_name="trainable_surface.central_ssl.trainer_version_prefix",
        ),
    )


def _read_required_config_str(source: Any, key: str, *, field_name: str) -> str:
    raw_value = getattr(source, key, None)
    if raw_value is None and hasattr(source, "get"):
        raw_value = source.get(key)
    value = "" if raw_value is None else str(raw_value).strip()
    if not value:
        raise ValueError(f"{field_name} is required.")
    return value
