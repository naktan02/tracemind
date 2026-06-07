"""Query SSL consistency family runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from methods.adaptation.query_text_views.data import DEFAULT_STRONG_VIEW_POLICY
from methods.adaptation.query_text_views.query_ssl_views import (
    build_query_ssl_unlabeled_dataloader,
)
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
from scripts.support.query_ssl_text_encoder.runtime_metrics import (
    run_with_training_runtime_metrics,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


@dataclass(frozen=True, slots=True)
class _CentralSslSurfaceRuntime:
    """중앙 SSL runner가 trainable_surface leaf에서 읽은 runtime callable."""

    surface_name: str
    model_builder: Callable[..., tuple[Any, Any, dict[str, Any]]]
    trainer: Callable[..., tuple[Any, list[dict[str, Any]], dict[str, Any]]]
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
        unlabeled_loader,
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
    model, history, best_selection_report, runtime_metrics = _train_query_ssl_context(
        cfg=cfg,
        context=context,
        unlabeled_loader=unlabeled_loader,
        algorithm=algorithm,
        surface_runtime=surface_runtime,
        max_train_steps=max_train_steps,
    )
    results = evaluate_query_ssl_run_context(
        model=model,
        eval_loaders=context.eval_loaders,
        categories=context.categories,
        device=context.training_device,
    )
    effective_extra_manifest = _build_query_ssl_extra_manifest(
        cfg=cfg,
        context=context,
        prepared_unlabeled_rows=prepared_unlabeled_rows,
        algorithm=algorithm,
        runtime_metrics=runtime_metrics,
        extra_manifest=extra_manifest,
    )
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
    Any,
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
    unlabeled_loader = _build_unlabeled_loader(
        cfg=cfg,
        descriptor=descriptor,
        context=context,
    )
    algorithm = descriptor.build_algorithm(build_query_ssl_method_parameters(cfg))
    return (
        prepared_unlabeled_rows,
        context,
        unlabeled_loader,
        algorithm,
        surface_runtime,
    )


def _train_query_ssl_context(
    *,
    cfg: Any,
    context: QuerySslRunContext,
    unlabeled_loader: Any,
    algorithm: QuerySslAlgorithm,
    surface_runtime: _CentralSslSurfaceRuntime,
    max_train_steps: int | None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """준비된 Query SSL context로 학습을 실행하고 runtime metric을 반환한다."""

    (
        (model, history, best_selection_report),
        runtime_metrics,
    ) = run_with_training_runtime_metrics(
        lambda: surface_runtime.trainer(
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
    return model, history, best_selection_report, runtime_metrics


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
        drop_last=bool(getattr(cfg, "drop_last_unlabeled_batches", False)),
    )


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
        trainer=load_configured_callable(
            _read_required_config_str(
                central_ssl,
                "trainer",
                field_name="trainable_surface.central_ssl.trainer",
            ),
            field_name="trainable_surface.central_ssl.trainer",
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
