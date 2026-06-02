"""SSL soft-target loss primitives."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .probability import compute_prob


def soft_cross_entropy_loss(*, logits: Tensor, targets: Tensor) -> Tensor:
    """USB CELoss soft-target branch와 같은 CE 평균."""

    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def probability_mse_loss(*, logits: Tensor, targets: Tensor) -> Tensor:
    """USB ConsistencyLoss(name='mse')와 같은 sample-wise MSE 평균."""

    probs = compute_prob(logits)
    return F.mse_loss(probs, targets, reduction="none").mean(dim=1).mean()
