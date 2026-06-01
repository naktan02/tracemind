"""FreeMatch method-local adaptive thresholding."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from methods.ssl.common import compute_prob


class FreeMatchThresholdingAlgorithm(Protocol):
    """FreeMatch threshold hook이 업데이트할 algorithm state surface."""

    use_quantile: bool
    clip_thresh: bool
    p_model: Tensor | None
    label_hist: Tensor | None
    time_p: Tensor | None


def replace_inf_to_zero(val: Tensor) -> Tensor:
    """USB FreeMatch의 inf-to-zero scaler 처리."""

    val[val == float("inf")] = 0.0
    return val


class FreeMatchThresholdingHook:
    """USB `FreeMatchThresholingHook`의 SAT thresholding state."""

    hook_name: str = "freematch_thresholding"

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
        self.p_model = torch.ones((self.num_classes,)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes,)) / self.num_classes
        self.time_p = self.p_model.mean()

    @torch.no_grad()
    def update(
        self,
        algorithm: FreeMatchThresholdingAlgorithm,
        probs_x_ulb: Tensor,
    ) -> None:
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1, keepdim=True)

        if algorithm.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(
                max_probs, 0.8
            )
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()

        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(
            max_idx.reshape(-1),
            minlength=self.p_model.shape[0],
        ).to(self.p_model.dtype)
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

        algorithm.p_model = self.p_model
        algorithm.label_hist = self.label_hist
        algorithm.time_p = self.time_p

    @torch.no_grad()
    def masking(
        self,
        algorithm: FreeMatchThresholdingAlgorithm,
        logits_x_ulb: Tensor,
        softmax_x_ulb: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        """USB FreeMatch SAT mask 수식."""

        del args, kwargs
        self._move_state_to_device(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = compute_prob(logits_x_ulb.detach())
        else:
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        return max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)

    def _move_state_to_device(self, device: torch.device) -> None:
        if self.p_model.device != device:
            self.p_model = self.p_model.to(device)
        if self.label_hist.device != device:
            self.label_hist = self.label_hist.to(device)
        if self.time_p.device != device:
            self.time_p = self.time_p.to(device)
