"""Query adaptation consistency SSL 알고리즘이 공유하는 공통 수식."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_prob(logits: Tensor) -> Tensor:
    """USB AlgorithmBase.compute_prob와 같은 softmax helper."""

    return torch.softmax(logits, dim=-1)
