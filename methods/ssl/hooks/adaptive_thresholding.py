"""Reusable SSL confidence threshold hooks."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class RelativeConfidenceThresholdAlgorithm(Protocol):
    """Relative confidence threshold hook이 읽는 algorithm surface."""

    p_cutoff: float


class RelativeConfidenceThresholdingHook:
    """labeled confidence 기준 relative threshold."""

    hook_name: str = "relative_confidence_threshold"

    @torch.no_grad()
    def masking(
        self,
        algorithm: RelativeConfidenceThresholdAlgorithm,
        *,
        probs_x_lb: Tensor,
        probs_x_ulb: Tensor,
    ) -> Tensor:
        max_probs_lb, _ = probs_x_lb.detach().max(dim=-1)
        p_cutoff = max_probs_lb.mean() * algorithm.p_cutoff
        max_probs_ulb, _ = probs_x_ulb.detach().max(dim=-1)
        return max_probs_ulb.ge(p_cutoff).to(max_probs_ulb.dtype)
