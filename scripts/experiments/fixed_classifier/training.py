"""Fixed classifier head 학습 helper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from methods.adaptation.common.selection_training_loop import (
    SelectionTrackedEpochResult,
    run_selection_tracked_training_loop,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

from .evaluation import evaluate_classifier


def build_label_index(
    rows: list[LabeledQueryRow],
) -> tuple[list[str], dict[str, int]]:
    categories = sorted({row["mapped_label_4"] for row in rows})
    return categories, {category: index for index, category in enumerate(categories)}


def labels_to_tensor(
    rows: list[LabeledQueryRow],
    label_to_index: dict[str, int],
) -> torch.Tensor:
    indices = [label_to_index[row["mapped_label_4"]] for row in rows]
    return torch.tensor(indices, dtype=torch.long)


def train_classifier_head(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    selection_features: torch.Tensor,
    selection_targets: torch.Tensor,
    categories: list[str],
    training_device: str,
    epochs: int,
    train_batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    """고정 임베딩 위 linear classifier head를 학습한다."""

    model = nn.Linear(train_features.shape[1], len(categories)).to(training_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    def train_epoch(_epoch: int) -> SelectionTrackedEpochResult:
        model.train()
        permutation = torch.randperm(len(train_targets))
        epoch_loss_total = 0.0

        for start in range(0, len(train_targets), train_batch_size):
            end = min(start + train_batch_size, len(train_targets))
            indices = permutation[start:end]
            batch_features = train_features[indices].to(training_device)
            batch_targets = train_targets[indices].to(training_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss_total += float(loss.item()) * len(indices)

        return SelectionTrackedEpochResult(
            train_loss_total=epoch_loss_total,
            train_loss_denominator=len(train_targets),
        )

    def evaluate_selection() -> dict[str, Any]:
        return evaluate_classifier(
            model=model,
            features=selection_features,
            targets=selection_targets,
            categories=categories,
            eval_batch_size=train_batch_size,
            device=training_device,
        )

    history, best_selection_report = run_selection_tracked_training_loop(
        model=model,
        epochs=epochs,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message=(
            "Classifier training did not produce a best checkpoint."
        ),
        log_epoch_summary=lambda message: print(message, flush=True),
    )
    return model, history, best_selection_report
