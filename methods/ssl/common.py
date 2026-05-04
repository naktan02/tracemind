"""Query adaptation consistency SSL 알고리즘이 공유하는 공통 수식."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def compute_prob(logits: Tensor) -> Tensor:
    """USB AlgorithmBase.compute_prob와 같은 softmax helper."""

    return torch.softmax(logits, dim=-1)


def consistency_cross_entropy_loss(
    *,
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """USB consistency_loss(name='ce')와 같은 masked CE 평균."""

    if targets.dtype in (torch.int32, torch.int64, torch.long):
        loss = F.cross_entropy(logits, targets, reduction="none")
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
    if mask is not None:
        loss = loss * mask
    return loss.mean()
