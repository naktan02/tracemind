"""LoRA + classifier scaffold 학습/평가 유틸리티."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
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
from methods.evaluation.classification_report import (
    build_classification_evaluation_report,
)
from methods.ssl.base import (
    QuerySslAlgorithm,
    QuerySslStepResult,
    configure_query_ssl_algorithm_dataset,
    configure_query_ssl_algorithm_training,
)
from shared.src.domain.services.classification_report import (
    safe_divide,
)

from .batching import (
    move_tensor_batch_to_device,
    next_cycling_batch,
)
from .modeling import LoraTextClassifier
from .optimizer_step import run_optimizer_loss_step
from .scalar_metrics import ScalarMetricAccumulator
from .step_budget import (
    remaining_effective_epochs,
    resolve_epoch_distributed_step_budget,
)


def set_seed(seed: int) -> None:
    """python/torch random seed를 맞춘다."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_classifier(
    *,
    model: LoraTextClassifier,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    categories: list[str],
    device: str,
) -> dict[str, Any]:
    """분류기 평가 지표를 계산한다."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    actual_labels: list[str] = []
    predicted_labels: list[str] = []
    true_probs: list[float] = []
    top_1_probs: list[float] = []
    margins: list[float] = []
    total_loss = 0.0
    total_rows = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
            predicted = top_indices[:, 0]
            true_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)

            batch_size = len(labels)
            total_rows += batch_size
            total_loss += float(loss.item()) * batch_size
            actual_labels.extend(categories[index] for index in labels.cpu().tolist())
            predicted_labels.extend(
                categories[index] for index in predicted.cpu().tolist()
            )
            true_probs.extend(true_probability.cpu().tolist())
            top_1_probs.extend(top_values[:, 0].cpu().tolist())
            margins.extend((top_values[:, 0] - top_values[:, 1]).cpu().tolist())

    return build_classification_evaluation_report(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        true_probs=true_probs,
        top_1_values=top_1_probs,
        margins=margins,
        total_loss=total_loss,
        total_rows=total_rows,
    )


def build_optimizer(
    *,
    model: LoraTextClassifier,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """LoRA 파라미터와 classifier 파라미터를 분리해 optimizer를 만든다."""

    classifier_params = []
    lora_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier"):
            classifier_params.append(parameter)
        else:
            lora_params.append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": classifier_params,
                "lr": classifier_learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )


def trainable_model_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """gradient update 대상 parameter만 반환한다."""

    return tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )


def train_classifier(
    *,
    model: LoraTextClassifier,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    selection_loader: DataLoader[dict[str, torch.Tensor]],
    categories: list[str],
    device: str,
    epochs: int,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    log_every_steps: int,
    max_train_steps: int | None = None,
    proximal_mu: float = 0.0,
) -> tuple[LoraTextClassifier, list[dict[str, Any]], dict[str, Any]]:
    """Supervised LoRA + classifier scaffold를 학습한다."""

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    trainable_parameters = trainable_model_parameters(model)
    fedprox = prepare_fedprox_regularizer(
        proximal_mu=proximal_mu,
        trainable_parameters=trainable_parameters,
    )
    criterion = nn.CrossEntropyLoss()
    step_budget = resolve_epoch_distributed_step_budget(
        epochs=int(epochs),
        full_epoch_steps=len(train_loader),
        max_train_steps=max_train_steps,
        invalid_max_steps_message="max_train_steps must be positive when provided.",
    )
    completed_steps = 0

    def train_epoch(epoch: int) -> SelectionTrackedEpochResult:
        nonlocal completed_steps

        model.train()
        epoch_loss_total = 0.0
        epoch_rows = 0
        train_iterator = iter(train_loader)
        epoch_steps = step_budget.remaining_epoch_steps(completed_steps)

        for step_index in range(1, epoch_steps + 1):
            batch, train_iterator = next_cycling_batch(
                loader=train_loader,
                iterator=train_iterator,
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            def compute_loss() -> torch.Tensor:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                return fedprox.add_to_loss(loss)

            loss = run_optimizer_loss_step(
                optimizer=optimizer,
                trainable_parameters=trainable_parameters,
                max_grad_norm=max_grad_norm,
                compute_loss=compute_loss,
            )

            batch_rows = len(labels)
            completed_steps += 1
            epoch_rows += batch_rows
            epoch_loss_total += float(loss.item()) * batch_rows

            if log_every_steps > 0 and step_index % log_every_steps == 0:
                running_loss = safe_divide(epoch_loss_total, epoch_rows)
                print(
                    f"[epoch={epoch} step={step_index}] "
                    f"running_train_loss={running_loss:.4f}",
                    flush=True,
                )

        return SelectionTrackedEpochResult(
            train_loss_total=epoch_loss_total,
            train_loss_denominator=epoch_rows,
        )

    def evaluate_selection() -> dict[str, Any]:
        return evaluate_classifier(
            model=model,
            dataloader=selection_loader,
            categories=categories,
            device=device,
        )

    history, best_selection_report = run_selection_tracked_training_loop(
        model=model,
        epochs=step_budget.effective_epochs,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message=(
            "LoRA classifier training did not produce a best checkpoint."
        ),
        log_epoch_summary=lambda message: print(message, flush=True),
    )
    return model, history, best_selection_report


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
    model: LoraTextClassifier,
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
    resume_checkpoint_path: str | Path | None = None,
    resume_checkpoint_output_dir: str | Path | None = None,
    resume_checkpoint_every_epochs: int = 0,
    proximal_mu: float = 0.0,
) -> tuple[LoraTextClassifier, list[dict[str, Any]], dict[str, Any]]:
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

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    trainable_parameters = trainable_model_parameters(model)
    fedprox = prepare_fedprox_regularizer(
        proximal_mu=proximal_mu,
        trainable_parameters=trainable_parameters,
    )

    resume_state = load_query_ssl_training_checkpoint(
        path=resume_checkpoint_path,
        model=model,
        optimizer=optimizer,
        algorithm=algorithm,
        categories=categories,
        device=device,
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

            def compute_total_loss() -> torch.Tensor:
                nonlocal step_output

                step_output = algorithm.compute_step(
                    model=model,
                    labeled_batch=labeled_batch,
                    unlabeled_batch=unlabeled_batch,
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
                trainable_parameters=trainable_parameters,
                max_grad_norm=max_grad_norm,
                compute_loss=compute_total_loss,
            )
            assert step_output is not None

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
        after_epoch=save_resume_checkpoint_after_epoch,
    )
    return model, history, best_selection_report
