"""SSL feature/target MixUp primitive."""

from __future__ import annotations

import torch
from torch import Tensor


def mixup_one_target(
    *,
    inputs: Tensor,
    targets: Tensor,
    alpha: float,
    bias_toward_primary: bool = True,
) -> tuple[Tensor, Tensor, float]:
    """USB `mixup_one_target(..., is_bias=True)`와 같은 MixUp을 수행한다."""

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same batch dimension.")
    if inputs.shape[0] <= 0:
        raise ValueError("inputs must not be empty.")
    if alpha < 0:
        raise ValueError("alpha must not be negative.")
    if alpha > 0:
        concentration = inputs.new_tensor(float(alpha))
        lam = float(torch.distributions.Beta(concentration, concentration).sample())
    else:
        lam = 1.0
    if bias_toward_primary:
        lam = max(lam, 1.0 - lam)

    index = torch.randperm(inputs.shape[0], device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    mixed_targets = lam * targets + (1.0 - lam) * targets[index]
    return mixed_inputs, mixed_targets, lam
