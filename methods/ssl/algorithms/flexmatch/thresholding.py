"""FlexMatch method-local adaptive thresholding."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Protocol

import torch
from torch import Tensor

from methods.ssl.common import compute_prob


class FlexMatchThresholdingAlgorithm(Protocol):
    """FlexMatch threshold hook이 읽는 algorithm surface."""

    p_cutoff: float


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
                for class_index in range(self.num_classes):
                    self.classwise_acc[class_index] = pseudo_counter[class_index] / max(
                        pseudo_counter.values()
                    )
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for class_index in range(self.num_classes):
                    self.classwise_acc[class_index] = pseudo_counter[class_index] / max(
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
        if self.selected_label.device != logits_x_ulb.device:
            self.selected_label = self.selected_label.to(logits_x_ulb.device)
        if self.classwise_acc.device != logits_x_ulb.device:
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
