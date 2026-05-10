"""LoRA + classifier scaffold 학습/평가 유틸리티."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.adaptation.common.classification_evaluation import (
    build_classification_evaluation_report,
)
from methods.adaptation.common.selection_training_loop import (
    SelectionTrackedEpochResult,
    run_selection_tracked_training_loop,
)
from methods.ssl.algorithms.fixmatch.fixmatch import (
    FixMatchAlgorithm,
    FixMatchConfig,
)
from methods.ssl.base import (
    QuerySslAlgorithm,
    configure_query_ssl_algorithm_training,
)
from shared.src.domain.services.classification_report import (
    safe_divide,
)

from .modeling import LoraTextClassifier


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
) -> tuple[LoraTextClassifier, list[dict[str, Any]], dict[str, Any]]:
    """Supervised LoRA + classifier scaffold를 학습한다."""

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    def train_epoch(epoch: int) -> SelectionTrackedEpochResult:
        model.train()
        epoch_loss_total = 0.0
        epoch_rows = 0

        for step_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_rows = len(labels)
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
        epochs=epochs,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message=(
            "LoRA classifier training did not produce a best checkpoint."
        ),
        log_epoch_summary=lambda message: print(message, flush=True),
    )
    return model, history, best_selection_report


def _next_batch(
    *,
    loader: DataLoader[dict[str, Any]],
    iterator: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        refreshed_iterator = iter(loader)
        return next(refreshed_iterator), refreshed_iterator


def _move_tensor_batch_to_device(
    *,
    batch: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _accumulate_scalar_tensors(
    *,
    sums: dict[str, float],
    values: Mapping[str, torch.Tensor],
) -> None:
    for name, value in values.items():
        sums[name] = sums.get(name, 0.0) + float(value.detach().item())


def _build_average_scalar_record(
    *,
    sums: dict[str, float],
    step_count: int,
) -> dict[str, float]:
    return {
        f"train_{name}": round(safe_divide(total, step_count), 6)
        for name, total in sums.items()
    }


def _format_running_scalars(
    *,
    total_loss_sum: float,
    component_sums: dict[str, float],
    metric_sums: dict[str, float],
    step_count: int,
) -> str:
    fields = [
        f"running_total_loss={safe_divide(total_loss_sum, step_count):.4f}",
    ]
    fields.extend(
        f"running_{name}={safe_divide(total, step_count):.4f}"
        for name, total in component_sums.items()
    )
    fields.extend(
        f"running_{name}={safe_divide(total, step_count):.4f}"
        for name, total in metric_sums.items()
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
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    log_every_steps: int,
    algorithm: QuerySslAlgorithm,
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
    epoch_steps = (
        max(len(train_loader), len(unlabeled_loader))
        if labeled_updates_enabled
        else len(unlabeled_loader)
    )
    configure_query_ssl_algorithm_training(
        algorithm,
        num_train_iter=max(1, int(epochs) * epoch_steps),
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )

    def train_epoch(epoch: int) -> SelectionTrackedEpochResult:
        model.train()
        step_total_loss_sum = 0.0
        step_component_sums: dict[str, float] = {}
        step_metric_sums: dict[str, float] = {}
        step_count = 0

        labeled_iterator = None if not labeled_updates_enabled else iter(train_loader)
        unlabeled_iterator = iter(unlabeled_loader)

        for step_index in range(1, epoch_steps + 1):
            if labeled_updates_enabled:
                assert labeled_iterator is not None
                labeled_batch, labeled_iterator = _next_batch(
                    loader=train_loader,
                    iterator=labeled_iterator,
                )
                labeled_batch = _move_tensor_batch_to_device(
                    batch=labeled_batch,
                    device=device,
                )
            else:
                labeled_batch = None
            unlabeled_batch, unlabeled_iterator = _next_batch(
                loader=unlabeled_loader,
                iterator=unlabeled_iterator,
            )
            unlabeled_batch = _move_tensor_batch_to_device(
                batch=unlabeled_batch,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            step_output = algorithm.compute_step(
                model=model,
                labeled_batch=labeled_batch,
                unlabeled_batch=unlabeled_batch,
            )
            step_output.total_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            step_count += 1
            step_total_loss_sum += float(step_output.total_loss.detach().item())
            _accumulate_scalar_tensors(
                sums=step_component_sums,
                values=step_output.loss_components,
            )
            _accumulate_scalar_tensors(
                sums=step_metric_sums,
                values=step_output.metrics,
            )

            if log_every_steps > 0 and step_index % log_every_steps == 0:
                print(
                    f"[epoch={epoch} step={step_index}] "
                    + _format_running_scalars(
                        total_loss_sum=step_total_loss_sum,
                        component_sums=step_component_sums,
                        metric_sums=step_metric_sums,
                        step_count=step_count,
                    ),
                    flush=True,
                )

        return SelectionTrackedEpochResult(
            train_loss_total=step_total_loss_sum,
            train_loss_denominator=step_count,
            extra_train_metrics={
                **_build_average_scalar_record(
                    sums=step_component_sums,
                    step_count=step_count,
                ),
                **_build_average_scalar_record(
                    sums=step_metric_sums,
                    step_count=step_count,
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

    history, best_selection_report = run_selection_tracked_training_loop(
        model=model,
        epochs=epochs,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message=(
            f"{algorithm.algorithm_name} training did not produce a best checkpoint."
        ),
        log_epoch_summary=lambda message: print(message, flush=True),
    )
    return model, history, best_selection_report


def train_fixmatch_classifier(
    *,
    model: LoraTextClassifier,
    train_loader: DataLoader[dict[str, Any]],
    unlabeled_loader: DataLoader[dict[str, Any]],
    selection_loader: DataLoader[dict[str, torch.Tensor]],
    categories: list[str],
    device: str,
    epochs: int,
    learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    log_every_steps: int,
    fixmatch_config: FixMatchConfig,
) -> tuple[LoraTextClassifier, list[dict[str, Any]], dict[str, Any]]:
    """기존 FixMatch 호출자를 위한 compatibility wrapper."""

    return train_query_ssl_classifier(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        selection_loader=selection_loader,
        categories=categories,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        log_every_steps=log_every_steps,
        algorithm=FixMatchAlgorithm(config=fixmatch_config),
    )
