"""Tensor-level SSL objective용 consistency loss hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F


class ConsistencyLossHook(Protocol):
    """strong-view logits와 weak-view target 사이의 consistency loss hook."""

    hook_name: str

    def compute_loss(
        self,
        *,
        logits: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """sample mask를 적용한 consistency loss를 계산한다."""


@dataclass(frozen=True, slots=True)
class CrossEntropyConsistencyLossHook:
    """USB consistency_loss(name='ce')와 같은 masked CE 평균."""

    hook_name: str = "cross_entropy_consistency"

    def compute_loss(
        self,
        *,
        logits: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return consistency_cross_entropy_loss(
            logits=logits,
            targets=targets,
            mask=mask,
        )


def consistency_cross_entropy_loss(
    *,
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """hard/soft target 모두 지원하는 masked consistency CE."""

    if targets.dtype in (torch.int32, torch.int64, torch.long):
        loss = F.cross_entropy(logits, targets, reduction="none")
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
    if mask is not None:
        loss = loss * mask
    return loss.mean()
