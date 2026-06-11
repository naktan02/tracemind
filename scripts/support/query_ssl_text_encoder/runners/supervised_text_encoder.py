"""중앙 supervised text encoder baseline 공통 runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.support.query_ssl_text_encoder.io.artifact_paths import (
    build_query_text_run_output_dir,
)
from scripts.support.query_ssl_text_encoder.io.supervised_epoch_checkpoints import (
    write_peft_supervised_epoch_checkpoint,
)
from scripts.support.query_ssl_text_encoder.result_utils import (
    extract_final_selection_report,
    merge_results_with_best_and_final,
)
from scripts.support.query_ssl_text_encoder.runtime_metrics import (
    run_with_training_runtime_metrics,
)
from scripts.support.query_ssl_text_encoder.text_encoder_run_context import (
    evaluate_text_encoder_run_context,
    prepare_text_encoder_run_context,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(frozen=True, slots=True)
class SupervisedTextEncoderTrainingRequest:
    """중앙 supervised text encoder 학습 함수가 받는 typed request surface."""

    model: Any
    train_loader: Any
    selection_loader: Any
    categories: list[str]
    device: str
    epochs: int
    max_train_steps: int | None
    learning_rate: float
    classifier_learning_rate: float
    weight_decay: float
    max_grad_norm: float
    log_every_steps: int
    after_epoch: Callable[[int, list[dict[str, Any]], dict[str, Any], Any], None] | None


def run_supervised_text_encoder_baseline(
    *,
    cfg,
    train_rows: list[LabeledQueryRow] | None,
    eval_rows_by_name: Mapping[str, list[LabeledQueryRow]] | None,
    selection_set_name: str | None,
    train_jsonl_ref: str | Path | None,
    eval_set_refs: Mapping[str, str | Path] | None,
    trainer_version_override: str | None,
    extra_manifest: Mapping[str, Any] | None,
    categories_override: list[str] | tuple[str, ...] | None,
    train_classifier_func: Callable[
        ..., tuple[Any, list[dict[str, Any]], dict[str, Any]]
    ],
    write_artifacts_func: Callable[..., dict[str, str]],
    model_builder: Callable[..., tuple[Any, Any, dict[str, Any]]],
    trainer_version_prefix: str,
) -> dict[str, str]:
    """중앙 supervised text encoder baseline 공통 실행 흐름."""

    context = prepare_text_encoder_run_context(
        cfg,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=selection_set_name,
        categories_override=categories_override,
        model_builder=model_builder,
        train_jsonl_ref=train_jsonl_ref,
        eval_set_refs=eval_set_refs,
        trainer_version_override=trainer_version_override,
        trainer_version_prefix=trainer_version_prefix,
    )
    max_train_steps = _resolve_max_train_steps(context.cfg)
    epoch_checkpoint_records: list[dict[str, str]] = []
    after_epoch = _build_epoch_checkpoint_callback(
        cfg=context.cfg,
        trainer_version=context.trainer_version,
        created_at=context.created_at,
        tokenizer=context.tokenizer,
        categories=context.categories,
        checkpoint_records=epoch_checkpoint_records,
    )
    (
        (model, history, best_selection_report),
        runtime_metrics,
    ) = run_with_training_runtime_metrics(
        lambda: _run_supervised_training(
            train_classifier_func=train_classifier_func,
            request=SupervisedTextEncoderTrainingRequest(
                model=context.model,
                train_loader=context.train_loader,
                selection_loader=context.selection_loader,
                categories=context.categories,
                device=context.training_device,
                epochs=int(context.cfg.epochs),
                max_train_steps=max_train_steps,
                learning_rate=float(context.cfg.learning_rate),
                classifier_learning_rate=float(context.cfg.classifier_learning_rate),
                weight_decay=float(context.cfg.weight_decay),
                max_grad_norm=float(context.cfg.max_grad_norm),
                log_every_steps=int(context.cfg.log_every_steps),
                after_epoch=after_epoch,
            ),
        ),
        training_example_count=_estimate_supervised_training_example_count(
            cfg=context.cfg,
            max_train_steps=max_train_steps,
            train_row_count=len(context.effective_train_rows),
        ),
        parameter_counts=context.backbone_summary["parameter_counts"],
        device=context.training_device,
    )

    results = evaluate_text_encoder_run_context(
        model=model,
        eval_loaders=context.eval_loaders,
        categories=context.categories,
        device=context.training_device,
    )
    final_selection_report = extract_final_selection_report(history)
    final_results = merge_results_with_best_and_final(
        results=results,
        selection_set=context.effective_selection_set,
        final_selection_report=final_selection_report,
    )

    effective_extra_manifest = dict(context.initial_checkpoint_manifest)
    effective_extra_manifest["runtime_metrics"] = runtime_metrics
    if epoch_checkpoint_records:
        effective_extra_manifest["epoch_checkpoint_policy"] = {
            "kind": str(getattr(context.cfg, "epoch_artifact_kind", "")),
            "every_epochs": int(getattr(context.cfg, "epoch_artifact_every_epochs", 0)),
        }
        effective_extra_manifest["epoch_checkpoints"] = epoch_checkpoint_records
    if extra_manifest:
        effective_extra_manifest.update(dict(extra_manifest))

    outputs = write_artifacts_func(
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


def _run_supervised_training(
    *,
    train_classifier_func: Callable[
        ..., tuple[Any, list[dict[str, Any]], dict[str, Any]]
    ],
    request: SupervisedTextEncoderTrainingRequest,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    kwargs: dict[str, object] = {
        "model": request.model,
        "train_loader": request.train_loader,
        "selection_loader": request.selection_loader,
        "categories": request.categories,
        "device": request.device,
        "epochs": request.epochs,
        "max_train_steps": request.max_train_steps,
        "learning_rate": request.learning_rate,
        "classifier_learning_rate": request.classifier_learning_rate,
        "weight_decay": request.weight_decay,
        "max_grad_norm": request.max_grad_norm,
        "log_every_steps": request.log_every_steps,
    }
    if request.after_epoch is not None:
        kwargs["after_epoch"] = request.after_epoch
    return train_classifier_func(**kwargs)


def _resolve_max_train_steps(cfg: Any) -> int | None:
    raw_value = getattr(cfg, "max_train_steps", None)
    if raw_value is None:
        return None
    return int(raw_value)


def _build_epoch_checkpoint_callback(
    *,
    cfg: Any,
    trainer_version: str,
    created_at: datetime,
    tokenizer: Any,
    categories: list[str],
    checkpoint_records: list[dict[str, str]],
) -> Callable[[int, list[dict[str, Any]], dict[str, Any], Any], None] | None:
    checkpoint_kind = str(getattr(cfg, "epoch_artifact_kind", "none") or "none")
    checkpoint_every_epochs = int(getattr(cfg, "epoch_artifact_every_epochs", 0) or 0)
    if checkpoint_kind == "none" or checkpoint_every_epochs <= 0:
        return None
    if checkpoint_kind != "peft_adapter_classifier":
        raise ValueError(f"Unsupported epoch_artifact_kind: {checkpoint_kind}")

    run_output_dir = build_query_text_run_output_dir(
        cfg=cfg,
        trainer_version=trainer_version,
        created_at=created_at,
    )
    checkpoint_root = run_output_dir / "checkpoints"

    def save_epoch_checkpoint(
        epoch: int,
        history: list[dict[str, Any]],
        best_checkpoint_state: dict[str, Any],
        model: Any,
    ) -> None:
        if epoch % checkpoint_every_epochs != 0:
            return
        checkpoint_records.append(
            write_peft_supervised_epoch_checkpoint(
                checkpoint_root=checkpoint_root,
                trainer_version=trainer_version,
                epoch=epoch,
                model=model,
                tokenizer=tokenizer,
                categories=categories,
                history=history,
                best_checkpoint_state=best_checkpoint_state,
            )
        )

    return save_epoch_checkpoint


def _estimate_supervised_training_example_count(
    *,
    cfg: Any,
    max_train_steps: int | None,
    train_row_count: int,
) -> int:
    if max_train_steps is None:
        return train_row_count * int(cfg.epochs)
    return int(max_train_steps) * int(cfg.train_batch_size)
