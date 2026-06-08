"""Text encoder + linear classifier supervised 학습/평가 유틸리티."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.adaptation.common.batching import next_cycling_batch
from methods.adaptation.common.optimizer_step import run_optimizer_loss_step
from methods.adaptation.common.selection_training_loop import (
    SelectionTrackedEpochResult,
    run_selection_tracked_training_loop,
)
from methods.adaptation.common.step_budget import resolve_epoch_distributed_step_budget
from methods.adaptation.local_objective_regularizers.fedprox import (
    prepare_fedprox_regularizer,
)
from methods.adaptation.text_encoder_classifier.modeling import (
    TextEncoderWithLinearHead,
)
from methods.evaluation.classification_report import (
    build_classification_evaluation_report,
)
from shared.src.domain.services.classification_report import safe_divide


def set_seed(seed: int) -> None:
    """python/torch random seed를 맞춘다."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_classifier(
    *,
    model: TextEncoderWithLinearHead,
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
    model: TextEncoderWithLinearHead,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """encoder-side trainable params와 classifier head params를 분리한다."""

    classifier_params = []
    encoder_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier"):
            classifier_params.append(parameter)
        else:
            encoder_params.append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": encoder_params,
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
    model: TextEncoderWithLinearHead,
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
    after_epoch: Callable[
        [int, list[dict[str, Any]], dict[str, Any], TextEncoderWithLinearHead],
        None,
    ]
    | None = None,
    best_checkpoint_error_message: str = (
        "Text encoder classifier training did not produce a best checkpoint."
    ),
) -> tuple[TextEncoderWithLinearHead, list[dict[str, Any]], dict[str, Any]]:
    """Supervised text encoder/head scaffold를 학습한다."""

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
            step=completed_steps,
        )

    def evaluate_selection() -> dict[str, Any]:
        return evaluate_classifier(
            model=model,
            dataloader=selection_loader,
            categories=categories,
            device=device,
        )

    def run_after_epoch(
        epoch: int,
        history: list[dict[str, Any]],
        best_checkpoint_state: dict[str, Any],
    ) -> None:
        if after_epoch is None:
            return
        after_epoch(epoch, history, best_checkpoint_state, model)

    history, best_selection_report = run_selection_tracked_training_loop(
        model=model,
        epochs=step_budget.effective_epochs,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message=best_checkpoint_error_message,
        log_epoch_summary=lambda message: print(message, flush=True),
        include_selection_report=True,
        after_epoch=run_after_epoch if after_epoch is not None else None,
    )
    return model, history, best_selection_report
