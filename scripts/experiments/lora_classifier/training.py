"""LoRA classifier 실험용 학습/평가 유틸리티."""

from __future__ import annotations

import copy
import random
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.classification_report import (
    build_confusion_matrix,
    safe_divide,
    summarize_per_category,
)
from .modeling import LoraTextClassifier


def set_seed(seed: int) -> None:
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
            predicted_labels.extend(categories[index] for index in predicted.cpu().tolist())
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
        raise RuntimeError("LoRA classifier training did not produce a best checkpoint.")

    model.load_state_dict(best_state_dict)
    return model, history, best_selection_report
