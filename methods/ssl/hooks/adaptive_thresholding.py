"""Stateful/adaptive SSL confidence threshold hooks."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
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


class RelativeConfidenceThresholdAlgorithm(Protocol):
    """Relative confidence threshold hook이 읽는 algorithm surface."""

    p_cutoff: float


class FlexMatchThresholdingAlgorithm(Protocol):
    """FlexMatch threshold hook이 읽는 algorithm surface."""

    p_cutoff: float


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


class RelativeConfidenceThresholdingHook:
    """AdaMatch-style labeled confidence 기준 relative threshold."""

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


class FlexMatchThresholdingHook:
    """USB `FlexMatchThresholdingHook`의 adaptive thresholding state."""

    hook_name: str = "flexmatch_thresholding"

    def __init__(
        self,
        *,
        ulb_dest_len: int,
        num_classes: int,
        thresh_warmup: bool = True,
    ) -> None:
        if ulb_dest_len <= 0:
            raise ValueError("ulb_dest_len must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.ulb_dest_len = int(ulb_dest_len)
        self.num_classes = int(num_classes)
        self.thresh_warmup = bool(thresh_warmup)
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long) * -1
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self, *args, **kwargs) -> None:
        """USB FlexMatch utils.py `update`와 같은 classwise_acc 갱신."""

        del args, kwargs
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.ulb_dest_len:
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(
                        pseudo_counter.values()
                    )
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(
                        wo_negative_one.values()
                    )

    @torch.no_grad()
    def masking(
        self,
        algorithm: FlexMatchThresholdingAlgorithm,
        logits_x_ulb: Tensor,
        idx_ulb: Tensor,
        softmax_x_ulb: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        """USB FlexMatch utils.py `masking`의 convex threshold 수식."""

        del args, kwargs
        if not self.selected_label.is_cuda and logits_x_ulb.is_cuda:
            self.selected_label = self.selected_label.to(logits_x_ulb.device)
        if not self.classwise_acc.is_cuda and logits_x_ulb.is_cuda:
            self.classwise_acc = self.classwise_acc.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = compute_prob(logits_x_ulb.detach())
        else:
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(
            algorithm.p_cutoff
            * (self.classwise_acc[max_idx] / (2.0 - self.classwise_acc[max_idx]))
        )
        select = max_probs.ge(algorithm.p_cutoff)
        mask = mask.to(max_probs.dtype)

        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
        self.update()

        return mask
