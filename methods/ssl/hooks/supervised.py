"""SSL supervised branch hooks."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from methods.ssl.base import TextBatchClassifier


def compute_labeled_cross_entropy_loss(
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
) -> Tensor | None:
    """labeled batch가 있으면 supervised CE loss를 계산한다."""

    if labeled_batch is None:
        return None
    logits_x_lb = model(
        input_ids=labeled_batch["input_ids"],
        attention_mask=labeled_batch["attention_mask"],
    )
    return F.cross_entropy(logits_x_lb, labeled_batch["labels"], reduction="mean")
