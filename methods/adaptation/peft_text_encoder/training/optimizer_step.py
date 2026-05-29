"""PEFT text encoder/head optimizer step lifecycle primitive."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn


def run_optimizer_loss_step(
    *,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: Iterable[nn.Parameter],
    max_grad_norm: float,
    compute_loss: Callable[[], Tensor],
) -> Tensor:
    """loss 계산 의미는 caller에 두고 optimizer lifecycle만 실행한다."""

    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss()
    loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            tuple(trainable_parameters),
            max_grad_norm,
        )
    optimizer.step()
    return loss
