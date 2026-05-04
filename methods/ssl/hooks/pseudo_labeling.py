"""Pseudo-label target generation hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PseudoLabelingConfig:
    """Pseudo-label target 생성 hook 입력."""

    use_hard_label: bool
    temperature: float = 1.0


class PseudoLabelingHook(Protocol):
    """weak-view prediction을 unlabeled target으로 바꾸는 SSL hook."""

    hook_name: str

    def generate_targets(
        self,
        *,
        probs_x_ulb_w: Tensor,
        config: PseudoLabelingConfig,
    ) -> Tensor:
        """weak-view probability로 hard/soft target을 만든다."""


@dataclass(frozen=True, slots=True)
class HardOrSoftPseudoLabelingHook:
    """USB `PseudoLabelingHook`의 hard/soft target 생성 규칙."""

    hook_name: str = "hard_or_soft_pseudo_labeling"

    def generate_targets(
        self,
        *,
        probs_x_ulb_w: Tensor,
        config: PseudoLabelingConfig,
    ) -> Tensor:
        return build_pseudo_label_from_probs(
            probs_x_ulb_w=probs_x_ulb_w,
            use_hard_label=config.use_hard_label,
            temperature=config.temperature,
        )


def build_pseudo_label_from_probs(
    *,
    probs_x_ulb_w: Tensor,
    use_hard_label: bool,
    temperature: float,
) -> Tensor:
    """USB `PseudoLabelingHook.gen_ulb_targets(..., softmax=False)` 경로."""

    del temperature
    detached_probs = probs_x_ulb_w.detach()
    if use_hard_label:
        return torch.argmax(detached_probs, dim=-1)
    return detached_probs
