"""USB weak/strong consistency objective 공통 helper."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from methods.ssl.base import QuerySslRequiredViews, TextBatchClassifier

USB_MULTIVIEW_REQUIRED_VIEWS = QuerySslRequiredViews(
    view_names=("text", "aug_0", "aug_1"),
    view_builder_name="usb_multiview",
)


def validate_usb_consistency_loaders(
    *,
    algorithm_name: str,
    train_loader_length: int,
    unlabeled_loader_length: int,
    supervised_loss_weight: float,
) -> None:
    """USB consistency algorithm의 labeled/unlabeled loader 전제를 검증한다."""

    if unlabeled_loader_length == 0:
        raise ValueError(f"{algorithm_name} unlabeled_loader must not be empty.")
    if supervised_loss_weight > 0 and train_loader_length == 0:
        raise ValueError(
            f"{algorithm_name} labeled train_loader must not be empty when "
            "supervised_loss_weight > 0."
        )


def compute_unlabeled_weak_strong_logits(
    *,
    model: TextBatchClassifier,
    unlabeled_batch: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    """USB weak/strong multiview batch에서 strong, weak logits를 계산한다."""

    logits_x_ulb_s = model(
        input_ids=unlabeled_batch["strong_input_ids"],
        attention_mask=unlabeled_batch["strong_attention_mask"],
    )
    with torch.no_grad():
        logits_x_ulb_w = model(
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )
    return logits_x_ulb_s, logits_x_ulb_w
