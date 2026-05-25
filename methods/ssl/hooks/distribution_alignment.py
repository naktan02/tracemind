"""SSL distribution alignment hooks."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class DistributionAlignmentStateReceiver(Protocol):
    """Distribution alignment hook이 algorithm-local mirror state를 갱신하는 surface."""

    p_model: Tensor | None
    p_target: Tensor | None


class AdaMatchDistAlignHook:
    """USB `DistAlignEMAHook(..., p_target_type='model')`의 EMA state."""

    hook_name: str = "adamatch_dist_align_ema"

    def __init__(
        self,
        *,
        num_classes: int,
        momentum: float = 0.999,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.m = float(momentum)
        self.p_model: Tensor | None = None
        self.p_target = torch.ones((self.num_classes,)) / self.num_classes

    @torch.no_grad()
    def dist_align(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor,
        algorithm: DistributionAlignmentStateReceiver | None = None,
    ) -> Tensor:
        """USB AdaMatch distribution alignment를 적용한 unlabeled probability."""

        self._move_state_to_device(probs_x_ulb.device)
        self.update_p(probs_x_ulb=probs_x_ulb, probs_x_lb=probs_x_lb)

        if self.p_model is None:  # pragma: no cover - defensive
            raise RuntimeError("AdaMatch distribution alignment state was not updated.")

        aligned_probs = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        aligned_probs = aligned_probs / aligned_probs.sum(dim=-1, keepdim=True)

        if algorithm is not None:
            algorithm.p_model = self.p_model
            algorithm.p_target = self.p_target
        return aligned_probs

    @torch.no_grad()
    def update_p(
        self,
        *,
        probs_x_ulb: Tensor,
        probs_x_lb: Tensor,
    ) -> None:
        """USB `DistAlignEMAHook.update_p`의 non-distributed EMA update."""

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model is None:
            self.p_model = probs_x_ulb.mean(dim=0)
        else:
            self.p_model = self.p_model * self.m + probs_x_ulb.mean(dim=0) * (
                1 - self.m
            )
        self.p_target = self.p_target * self.m + probs_x_lb.mean(dim=0) * (1 - self.m)

    def _move_state_to_device(self, device: torch.device) -> None:
        if self.p_target.device != device:
            self.p_target = self.p_target.to(device)
        if self.p_model is not None and self.p_model.device != device:
            self.p_model = self.p_model.to(device)
