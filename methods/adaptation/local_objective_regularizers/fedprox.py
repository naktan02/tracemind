"""FedProx local proximal regularizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class FedProxReferenceSnapshot:
    """round 시작 global trainable parameter snapshot."""

    parameters: tuple[torch.Tensor, ...]


def snapshot_trainable_parameters(
    parameters: tuple[nn.Parameter, ...],
) -> FedProxReferenceSnapshot:
    """FedProx 기준점으로 쓸 trainable parameter snapshot을 만든다."""

    if not parameters:
        raise ValueError("FedProx requires at least one trainable parameter.")
    return FedProxReferenceSnapshot(
        parameters=tuple(parameter.detach().clone() for parameter in parameters)
    )


def compute_fedprox_proximal_loss(
    *,
    trainable_parameters: tuple[nn.Parameter, ...],
    reference_snapshot: FedProxReferenceSnapshot,
) -> torch.Tensor:
    """FedProx proximal term의 unscaled L2 제곱합을 계산한다."""

    global_parameters = reference_snapshot.parameters
    if len(trainable_parameters) != len(global_parameters):
        raise ValueError("FedProx parameter snapshot length mismatch.")
    if not trainable_parameters:
        raise ValueError("FedProx requires at least one trainable parameter.")
    proximal_loss = trainable_parameters[0].new_tensor(0.0)
    for parameter, global_parameter in zip(trainable_parameters, global_parameters):
        if parameter.shape != global_parameter.shape:
            raise ValueError("FedProx parameter snapshot shape mismatch.")
        proximal_loss = proximal_loss + torch.sum(
            (parameter - global_parameter.to(parameter.device)) ** 2
        )
    return proximal_loss


def compute_scaled_fedprox_loss(
    *,
    trainable_parameters: tuple[nn.Parameter, ...],
    reference_snapshot: FedProxReferenceSnapshot,
    proximal_mu: float,
) -> torch.Tensor:
    """원 논문/구현의 mu/2 * ||w - w_global||^2 항을 계산한다."""

    return 0.5 * proximal_mu * compute_fedprox_proximal_loss(
        trainable_parameters=trainable_parameters,
        reference_snapshot=reference_snapshot,
    )


def validate_proximal_mu(proximal_mu: float) -> float:
    """FedProx proximal coefficient를 정규화한다."""

    if isinstance(proximal_mu, bool):
        raise ValueError("proximal_mu must not be bool.")
    normalized = float(proximal_mu)
    if normalized < 0.0:
        raise ValueError("proximal_mu must be non-negative.")
    return normalized
