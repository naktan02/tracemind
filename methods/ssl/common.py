"""Query adaptation consistency SSL 알고리즘이 공유하는 공통 수식."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def compute_prob(logits: Tensor) -> Tensor:
    """USB AlgorithmBase.compute_prob와 같은 softmax helper."""

    return torch.softmax(logits, dim=-1)


def build_fixed_threshold_mask(
    *,
    probs_x_ulb_w: Tensor,
    p_cutoff: float,
) -> Tensor:
    """USB FixedThresholdingHook.masking과 같은 전역 confidence mask."""

    max_probs, _ = torch.max(probs_x_ulb_w, dim=-1)
    return max_probs.ge(p_cutoff).to(max_probs.dtype)


def build_pseudo_label_from_probs(
    *,
    probs_x_ulb_w: Tensor,
    use_hard_label: bool,
    temperature: float,
) -> Tensor:
    """USB PseudoLabelingHook의 FixMatch 호출 경로와 같은 target 생성."""

    del temperature
    detached_probs = probs_x_ulb_w.detach()
    if use_hard_label:
        return torch.argmax(detached_probs, dim=-1)
    return detached_probs


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
