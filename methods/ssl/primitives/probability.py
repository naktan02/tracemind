"""SSL probability transform primitives."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_prob(logits: Tensor) -> Tensor:
    """USB AlgorithmBase.compute_prob와 같은 softmax helper."""

    return torch.softmax(logits, dim=-1)


def sharpen_probabilities(probs: Tensor, *, temperature: float) -> Tensor:
    """MixMatch류 temperature sharpening 수식."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    sharpened = probs ** (1.0 / float(temperature))
    return sharpened / sharpened.sum(dim=-1, keepdim=True).clamp_min(1e-12)
