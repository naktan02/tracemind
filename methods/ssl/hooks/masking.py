"""Pseudo-label mask hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class MaskingHook(Protocol):
    """unlabeled consistency loss에 적용할 sample mask를 만드는 SSL hook."""

    hook_name: str

    def build_mask(
        self,
        *,
        probs_x_ulb_w: Tensor,
        p_cutoff: float,
    ) -> Tensor:
        """weak-view probability로 sample별 mask를 만든다."""


@dataclass(frozen=True, slots=True)
class FixedThresholdMaskingHook:
    """USB `FixedThresholdingHook`과 같은 전역 confidence mask."""

    hook_name: str = "fixed_threshold"

    def build_mask(
        self,
        *,
        probs_x_ulb_w: Tensor,
        p_cutoff: float,
    ) -> Tensor:
        return build_fixed_threshold_mask(
            probs_x_ulb_w=probs_x_ulb_w,
            p_cutoff=p_cutoff,
        )


def build_fixed_threshold_mask(
    *,
    probs_x_ulb_w: Tensor,
    p_cutoff: float,
) -> Tensor:
    """USB `FixedThresholdingHook.masking`과 같은 전역 confidence mask."""

    max_probs, _ = torch.max(probs_x_ulb_w, dim=-1)
    return max_probs.ge(p_cutoff).to(max_probs.dtype)
