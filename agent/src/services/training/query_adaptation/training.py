"""Query adaptation supervised baseline용 학습/평가 유틸리티."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from shared.src.domain.services.classification_report import (
    build_confusion_matrix,
    safe_divide,
    summarize_per_category,
)

from .algorithms.fixmatch import FixMatchConfig, compute_fixmatch_step
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

    correct = sum(
        1
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    confusion_matrix = build_confusion_matrix(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )
    per_category = summarize_per_category(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        primary_values=true_probs,
        top_1_values=top_1_probs,
        margins=margins,
        primary_metric_key="mean_true_label_probability",
        top_1_metric_key="mean_top_1_probability",
    )
    return {
        "rows_total": total_rows,
        "loss": round(safe_divide(total_loss, total_rows), 6),
        "accuracy_top_1": round(safe_divide(correct, total_rows), 6),
        "correct_top_1": correct,
        "mean_true_label_probability": round(
            safe_divide(sum(true_probs), len(true_probs)),
            6,
        ),
        "mean_top_1_probability": round(
            safe_divide(sum(top_1_probs), len(top_1_probs)),
            6,
        ),
        "mean_margin_top1_top2": round(
            safe_divide(sum(margins), len(margins)),
            6,
        ),
        "confusion_matrix": confusion_matrix,
        "per_category": per_category,
    }


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
    """Supervised LoRA + classifier baseline을 학습한다."""

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selection_report: dict[str, Any] | None = None
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
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

        selection_report = evaluate_classifier(
            model=model,
            dataloader=selection_loader,
            categories=categories,
            device=device,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(safe_divide(epoch_loss_total, epoch_rows), 6),
            "selection_loss": selection_report["loss"],
            "selection_accuracy_top_1": selection_report["accuracy_top_1"],
        }
        history.append(epoch_record)
        print(
            f"[epoch={epoch}] "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"selection_loss={epoch_record['selection_loss']:.4f} "
            f"selection_accuracy={epoch_record['selection_accuracy_top_1']:.4f}",
            flush=True,
        )

        accuracy = float(selection_report["accuracy_top_1"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
            best_selection_report = selection_report

    if best_state_dict is None or best_selection_report is None:
        raise RuntimeError(
            "LoRA classifier training did not produce a best checkpoint."
        )

    model.load_state_dict(best_state_dict)
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
    """USB FixMatch core를 epoch-based query adaptation scaffold에 얹어 학습한다."""

    if len(unlabeled_loader) == 0:
        raise ValueError("FixMatch unlabeled_loader must not be empty.")
    labeled_updates_enabled = (
        fixmatch_config.supervised_loss_weight > 0 and len(train_loader) > 0
    )
    if fixmatch_config.supervised_loss_weight > 0 and len(train_loader) == 0:
        raise ValueError(
            "FixMatch labeled train_loader must not be empty when "
            "supervised_loss_weight > 0."
        )

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selection_report: dict[str, Any] | None = None
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        step_total_loss_sum = 0.0
        step_sup_loss_sum = 0.0
        step_unsup_loss_sum = 0.0
        step_util_ratio_sum = 0.0
        step_count = 0

        labeled_iterator = None if not labeled_updates_enabled else iter(train_loader)
        unlabeled_iterator = iter(unlabeled_loader)
        epoch_steps = (
            max(len(train_loader), len(unlabeled_loader))
            if labeled_updates_enabled
            else len(unlabeled_loader)
        )

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
            step_output = compute_fixmatch_step(
                model=model,
                labeled_batch=labeled_batch,
                unlabeled_batch=unlabeled_batch,
                config=fixmatch_config,
            )
            step_output.total_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            step_count += 1
            step_total_loss_sum += float(step_output.total_loss.item())
            step_sup_loss_sum += float(step_output.sup_loss.item())
            step_unsup_loss_sum += float(step_output.unsup_loss.item())
            step_util_ratio_sum += float(step_output.util_ratio.item())

            if log_every_steps > 0 and step_index % log_every_steps == 0:
                print(
                    f"[epoch={epoch} step={step_index}] "
                    f"running_total_loss="
                    f"{safe_divide(step_total_loss_sum, step_count):.4f} "
                    f"running_sup_loss="
                    f"{safe_divide(step_sup_loss_sum, step_count):.4f} "
                    f"running_unsup_loss="
                    f"{safe_divide(step_unsup_loss_sum, step_count):.4f} "
                    f"running_util_ratio="
                    f"{safe_divide(step_util_ratio_sum, step_count):.4f}",
                    flush=True,
                )

        selection_report = evaluate_classifier(
            model=model,
            dataloader=selection_loader,
            categories=categories,
            device=device,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(safe_divide(step_total_loss_sum, step_count), 6),
            "train_sup_loss": round(safe_divide(step_sup_loss_sum, step_count), 6),
            "train_unsup_loss": round(safe_divide(step_unsup_loss_sum, step_count), 6),
            "train_util_ratio": round(safe_divide(step_util_ratio_sum, step_count), 6),
            "selection_loss": selection_report["loss"],
            "selection_accuracy_top_1": selection_report["accuracy_top_1"],
        }
        history.append(epoch_record)
        print(
            f"[epoch={epoch}] "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"train_sup_loss={epoch_record['train_sup_loss']:.4f} "
            f"train_unsup_loss={epoch_record['train_unsup_loss']:.4f} "
            f"train_util_ratio={epoch_record['train_util_ratio']:.4f} "
            f"selection_loss={epoch_record['selection_loss']:.4f} "
            f"selection_accuracy={epoch_record['selection_accuracy_top_1']:.4f}",
            flush=True,
        )

        accuracy = float(selection_report["accuracy_top_1"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
            best_selection_report = selection_report

    if best_state_dict is None or best_selection_report is None:
        raise RuntimeError("FixMatch training did not produce a best checkpoint.")

    model.load_state_dict(best_state_dict)
    return model, history, best_selection_report


__all__ = [
    "build_optimizer",
    "evaluate_classifier",
    "set_seed",
    "train_classifier",
    "train_fixmatch_classifier",
]
