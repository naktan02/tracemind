"""PEFT text encoder/head Query SSL 학습/평가 유틸리티."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from methods.adaptation.common.query_ssl_training_resume import (
    load_query_ssl_training_checkpoint,
    save_query_ssl_training_checkpoint,
)
from methods.adaptation.common.selection_training_loop import (
    SelectionTrackedEpochResult,
    run_selection_tracked_training_loop,
)
from methods.adaptation.local_objective_regularizers.fedprox import (
    prepare_fedprox_regularizer,
)
from methods.adaptation.text_encoder_classifier import training as _text_training
from methods.ssl.base import (
    QuerySslAlgorithm,
    QuerySslStepContext,
    QuerySslStepResult,
)
from methods.ssl.runtime.lifecycle import (
    after_query_ssl_optimizer_step,
    compute_query_ssl_algorithm_step,
    configure_query_ssl_algorithm_batching,
    configure_query_ssl_algorithm_dataset,
    configure_query_ssl_algorithm_initial_selection_loss,
    configure_query_ssl_algorithm_labeled_dataset,
    configure_query_ssl_algorithm_labeled_distribution,
    configure_query_ssl_algorithm_model,
    configure_query_ssl_algorithm_training,
)
from methods.ssl.state import load_query_ssl_algorithm_state
from shared.src.domain.services.classification_report import (
    safe_divide,
)

from .batching import (
    move_tensor_batch_to_device,
    next_cycling_batch,
)
from .modeling import PeftTextEncoderWithLinearHead
from .optimizer_step import run_optimizer_loss_step
from .scalar_metrics import ScalarMetricAccumulator
from .ssl_model_extensions import (
    build_peft_query_ssl_model_extensions,
    set_peft_query_ssl_auxiliary_modules_train,
)
from .step_budget import (
    remaining_effective_epochs,
    resolve_epoch_distributed_step_budget,
)

set_seed = _text_training.set_seed
evaluate_classifier = _text_training.evaluate_classifier
build_optimizer = _text_training.build_optimizer
trainable_model_parameters = _text_training.trainable_model_parameters
train_classifier = _text_training.train_classifier


def _format_running_scalars(
    *,
    total_loss_sum: float,
    component_metrics: ScalarMetricAccumulator,
    step_metrics: ScalarMetricAccumulator,
    step_count: int,
) -> str:
    fields = [
        f"running_total_loss={safe_divide(total_loss_sum, step_count):.4f}",
    ]
    fields.extend(
        component_metrics.running_fields(
            denominator=step_count,
            key_prefix="running_",
        )
    )
    fields.extend(
        step_metrics.running_fields(
            denominator=step_count,
            key_prefix="running_",
        )
    )
    return " ".join(fields)


def train_query_ssl_classifier(
    *,
    model: PeftTextEncoderWithLinearHead,
    train_loader: DataLoader[dict[str, Any]],
    unlabeled_loader: DataLoader[dict[str, Any]],
    selection_loader: DataLoader[dict[str, torch.Tensor]],
    categories: list[str],
    device: str,
    epochs: int,
    max_train_steps: int | None,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    log_every_steps: int,
    algorithm: QuerySslAlgorithm,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
    resume_checkpoint_path: str | Path | None = None,
    resume_checkpoint_output_dir: str | Path | None = None,
    resume_checkpoint_every_epochs: int = 0,
    proximal_mu: float = 0.0,
) -> tuple[PeftTextEncoderWithLinearHead, list[dict[str, Any]], dict[str, Any]]:
    """Query SSL algorithm을 epoch-based query adaptation scaffold에 얹어 학습한다."""

    algorithm.validate_loaders(
        train_loader_length=len(train_loader),
        unlabeled_loader_length=len(unlabeled_loader),
    )
    labeled_updates_enabled = algorithm.uses_labeled_batches and len(train_loader) > 0
    if algorithm.uses_labeled_batches and len(train_loader) == 0:
        raise ValueError(
            f"{algorithm.algorithm_name} labeled train_loader must not be empty "
            "when the algorithm uses labeled batches."
        )
    configure_query_ssl_algorithm_dataset(
        algorithm,
        num_classes=len(categories),
        unlabeled_row_count=len(unlabeled_loader.dataset),
    )
    configure_query_ssl_algorithm_labeled_dataset(
        algorithm,
        labeled_row_count=len(train_loader.dataset),
    )
    labeled_class_distribution = _loader_labeled_class_distribution(
        train_loader,
        num_classes=len(categories),
    )
    if labeled_class_distribution is not None:
        configure_query_ssl_algorithm_labeled_distribution(
            algorithm,
            class_distribution=labeled_class_distribution,
        )
    if initial_query_ssl_algorithm_state and resume_checkpoint_path is not None:
        raise ValueError(
            "initial_query_ssl_algorithm_state and resume_checkpoint_path cannot "
            "both be provided."
        )
    full_epoch_steps = (
        max(len(train_loader), len(unlabeled_loader))
        if labeled_updates_enabled
        else len(unlabeled_loader)
    )
    step_budget = resolve_epoch_distributed_step_budget(
        epochs=int(epochs),
        full_epoch_steps=full_epoch_steps,
        max_train_steps=max_train_steps,
        invalid_max_steps_message="max_train_steps must be positive when provided.",
    )
    configure_query_ssl_algorithm_training(
        algorithm,
        num_train_iter=max(1, step_budget.total_train_steps),
    )
    configure_query_ssl_algorithm_batching(
        algorithm,
        labeled_batch_size=(
            _require_loader_batch_size(
                train_loader,
                loader_name="train_loader",
            )
            if labeled_updates_enabled
            else 0
        ),
        unlabeled_batch_size=_require_loader_batch_size(
            unlabeled_loader,
            loader_name="unlabeled_loader",
        ),
    )
    configure_query_ssl_algorithm_model(
        algorithm,
        model=model,
        device=torch.device(device),
    )
    if initial_query_ssl_algorithm_state:
        load_query_ssl_algorithm_state(
            algorithm,
            initial_query_ssl_algorithm_state,
        )

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    model_extensions = build_peft_query_ssl_model_extensions(
        algorithm=algorithm,
        model=model,
        device=device,
    )
    if model_extensions.auxiliary_trainable_parameters:
        optimizer.add_param_group(
            {
                "params": list(model_extensions.auxiliary_trainable_parameters),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    trainable_parameters = trainable_model_parameters(model)
    optimizer_step_parameters = (
        trainable_parameters + model_extensions.auxiliary_trainable_parameters
    )
    fedprox = prepare_fedprox_regularizer(
        proximal_mu=proximal_mu,
        trainable_parameters=trainable_parameters,
    )
    if resume_checkpoint_path is None and bool(
        getattr(algorithm, "needs_initial_selection_loss", False)
    ):
        warmup_steps = _resolve_initial_selection_warmup_steps(algorithm)
        if warmup_steps > 0:
            _run_initial_selection_supervised_warmup(
                model=model,
                train_loader=train_loader,
                device=device,
                max_steps=warmup_steps,
                optimizer=optimizer,
                trainable_parameters=optimizer_step_parameters,
                max_grad_norm=max_grad_norm,
                fedprox=fedprox,
                log_every_steps=log_every_steps,
                algorithm_name=algorithm.algorithm_name,
            )
        initial_selection_report = evaluate_classifier(
            model=model,
            dataloader=selection_loader,
            categories=categories,
            device=device,
        )
        configure_query_ssl_algorithm_initial_selection_loss(
            algorithm,
            selection_loss=float(initial_selection_report["loss"]),
        )

    resume_state = load_query_ssl_training_checkpoint(
        path=resume_checkpoint_path,
        model=model,
        optimizer=optimizer,
        algorithm=algorithm,
        categories=categories,
        device=device,
        auxiliary_modules=model_extensions.auxiliary_modules,
    )
    completed_steps = resume_state.completed_steps
    initial_history = resume_state.history
    initial_best_checkpoint_state = resume_state.best_checkpoint_state
    remaining_steps = max(0, step_budget.total_train_steps - completed_steps)
    effective_epochs = remaining_effective_epochs(
        epochs=int(epochs),
        remaining_steps=remaining_steps,
        steps_per_epoch_budget=step_budget.steps_per_epoch_budget,
    )
    checkpoint_every_epochs = int(resume_checkpoint_every_epochs)
    if checkpoint_every_epochs < 0:
        raise ValueError("resume_checkpoint_every_epochs must not be negative.")
    checkpoint_output_dir = (
        None
        if resume_checkpoint_output_dir is None
        or not str(resume_checkpoint_output_dir).strip()
        else Path(resume_checkpoint_output_dir)
    )

    def train_epoch(epoch: int) -> SelectionTrackedEpochResult:
        nonlocal completed_steps

        model.train()
        set_peft_query_ssl_auxiliary_modules_train(
            model_extensions,
            training=True,
        )
        step_total_loss_sum = 0.0
        component_metrics = ScalarMetricAccumulator()
        step_metrics = ScalarMetricAccumulator()
        step_count = 0

        labeled_iterator = None if not labeled_updates_enabled else iter(train_loader)
        unlabeled_iterator = iter(unlabeled_loader)

        epoch_steps = step_budget.remaining_epoch_steps(completed_steps)
        for step_index in range(1, epoch_steps + 1):
            if labeled_updates_enabled:
                assert labeled_iterator is not None
                labeled_batch, labeled_iterator = next_cycling_batch(
                    loader=train_loader,
                    iterator=labeled_iterator,
                )
                labeled_batch = move_tensor_batch_to_device(
                    batch=labeled_batch,
                    device=device,
                )
            else:
                labeled_batch = None
            unlabeled_batch, unlabeled_iterator = next_cycling_batch(
                loader=unlabeled_loader,
                iterator=unlabeled_iterator,
            )
            unlabeled_batch = move_tensor_batch_to_device(
                batch=unlabeled_batch,
                device=device,
            )

            step_output: QuerySslStepResult | None = None
            step_context = QuerySslStepContext(
                epoch_index=epoch,
                step_index=step_index,
                global_step=completed_steps + 1,
                total_train_steps=step_budget.total_train_steps,
                num_classes=len(categories),
                device=torch.device(device),
            )

            def compute_total_loss() -> torch.Tensor:
                nonlocal step_output

                step_output = compute_query_ssl_algorithm_step(
                    algorithm,
                    model=model,
                    labeled_batch=labeled_batch,
                    unlabeled_batch=unlabeled_batch,
                    step_context=step_context,
                )
                total_loss = step_output.total_loss
                if fedprox.enabled:
                    fedprox_loss = fedprox.proximal_loss()
                    total_loss = total_loss + fedprox_loss
                    component_metrics.add_tensor(
                        "fedprox_proximal_loss",
                        fedprox_loss,
                    )
                return total_loss

            total_loss = run_optimizer_loss_step(
                optimizer=optimizer,
                trainable_parameters=optimizer_step_parameters,
                max_grad_norm=max_grad_norm,
                compute_loss=compute_total_loss,
            )
            assert step_output is not None
            after_query_ssl_optimizer_step(
                algorithm,
                model=model,
                step_context=step_context,
            )

            step_count += 1
            completed_steps += 1
            step_total_loss_sum += float(total_loss.detach().item())
            component_metrics.add_tensor_mapping(step_output.loss_components)
            step_metrics.add_tensor_mapping(step_output.metrics)

            if log_every_steps > 0 and step_index % log_every_steps == 0:
                print(
                    f"[epoch={epoch} step={step_index}] "
                    + _format_running_scalars(
                        total_loss_sum=step_total_loss_sum,
                        component_metrics=component_metrics,
                        step_metrics=step_metrics,
                        step_count=step_count,
                    ),
                    flush=True,
                )

        return SelectionTrackedEpochResult(
            train_loss_total=step_total_loss_sum,
            train_loss_denominator=step_count,
            step=completed_steps,
            extra_train_metrics={
                **component_metrics.average_record(
                    denominator=step_count,
                    key_prefix="train_",
                ),
                **step_metrics.average_record(
                    denominator=step_count,
                    key_prefix="train_",
                ),
            },
        )

    def evaluate_selection() -> dict[str, Any]:
        return evaluate_classifier(
            model=model,
            dataloader=selection_loader,
            categories=categories,
            device=device,
        )

    def save_resume_checkpoint_after_epoch(
        epoch: int,
        history: list[dict[str, Any]],
        best_checkpoint_state: dict[str, Any],
    ) -> None:
        if checkpoint_output_dir is None or checkpoint_every_epochs <= 0:
            return
        if (
            epoch % checkpoint_every_epochs != 0
            and completed_steps < step_budget.total_train_steps
        ):
            return
        save_query_ssl_training_checkpoint(
            checkpoint_output_dir=checkpoint_output_dir,
            algorithm=algorithm,
            model=model,
            optimizer=optimizer,
            completed_steps=completed_steps,
            total_train_steps=step_budget.total_train_steps,
            history=history,
            best_checkpoint_state=best_checkpoint_state,
            categories=categories,
            auxiliary_modules=model_extensions.auxiliary_modules,
        )

    history, best_selection_report = run_selection_tracked_training_loop(
        model=model,
        epochs=effective_epochs,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message=(
            f"{algorithm.algorithm_name} training did not produce a best checkpoint."
        ),
        log_epoch_summary=lambda message: print(message, flush=True),
        initial_history=initial_history,
        initial_best_checkpoint_state=initial_best_checkpoint_state,
        include_selection_report=True,
        after_epoch=save_resume_checkpoint_after_epoch,
    )
    return model, history, best_selection_report


def _resolve_initial_selection_warmup_steps(algorithm: QuerySslAlgorithm) -> int:
    raw_steps = getattr(algorithm, "initial_selection_warmup_steps", 0)
    warmup_steps = int(raw_steps() if callable(raw_steps) else raw_steps)
    if warmup_steps < 0:
        raise ValueError(
            f"{algorithm.algorithm_name} initial_selection_warmup_steps must not "
            "be negative."
        )
    return warmup_steps


def _run_initial_selection_supervised_warmup(
    *,
    model: PeftTextEncoderWithLinearHead,
    train_loader: DataLoader[dict[str, Any]],
    device: str,
    max_steps: int,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: tuple[torch.nn.Parameter, ...],
    max_grad_norm: float,
    fedprox: Any,
    log_every_steps: int,
    algorithm_name: str,
) -> None:
    """DASH처럼 rho_init 계산 전 labeled-only warmup을 수행한다."""

    if max_steps <= 0:
        return
    model.train()
    train_iterator = iter(train_loader)
    running_loss_total = 0.0
    for step_index in range(1, max_steps + 1):
        batch, train_iterator = next_cycling_batch(
            loader=train_loader,
            iterator=train_iterator,
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        def compute_loss() -> torch.Tensor:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return fedprox.add_to_loss(loss)

        loss = run_optimizer_loss_step(
            optimizer=optimizer,
            trainable_parameters=trainable_parameters,
            max_grad_norm=max_grad_norm,
            compute_loss=compute_loss,
        )
        running_loss_total += float(loss.detach().item())
        if log_every_steps > 0 and step_index % log_every_steps == 0:
            running_loss = safe_divide(running_loss_total, step_index)
            print(
                f"[{algorithm_name} warmup step={step_index}] "
                f"running_sup_loss={running_loss:.4f}",
                flush=True,
            )


def _require_loader_batch_size(
    loader: DataLoader[dict[str, Any]],
    *,
    loader_name: str,
) -> int:
    batch_size = getattr(loader, "batch_size", None)
    if batch_size is None:
        raise ValueError(f"{loader_name}.batch_size is required.")
    normalized = int(batch_size)
    if normalized <= 0:
        raise ValueError(f"{loader_name}.batch_size must be positive.")
    return normalized


def _loader_labeled_class_distribution(
    loader: DataLoader[dict[str, Any]],
    *,
    num_classes: int,
) -> torch.Tensor | None:
    dataset = getattr(loader, "dataset", None)
    label_histogram = getattr(dataset, "label_histogram", None)
    if not callable(label_histogram):
        return None
    counts = label_histogram(num_classes=num_classes)
    if not isinstance(counts, torch.Tensor):
        raise TypeError("label_histogram must return a torch.Tensor.")
    if counts.shape != (num_classes,):
        raise ValueError("label_histogram shape must match num_classes.")
    total = counts.sum()
    if float(total.item()) <= 0:
        raise ValueError("label_histogram must have positive mass.")
    return counts.to(dtype=torch.float32) / total
