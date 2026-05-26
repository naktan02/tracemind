"""LoRA-classifier training FedProx helper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from methods.adaptation.local_objective_regularizers.fedprox import (
    FedProxReferenceSnapshot,
    compute_scaled_fedprox_loss,
    snapshot_trainable_parameters,
    validate_proximal_mu,
)


@dataclass(frozen=True, slots=True)
class LoraTrainingFedProxState:
    """LoRA training loop에서 FedProx 기준 snapshot과 적용 조건을 보존한다."""

    proximal_mu: float
    trainable_parameters: tuple[nn.Parameter, ...]
    reference_snapshot: FedProxReferenceSnapshot | None

    @property
    def enabled(self) -> bool:
        return self.proximal_mu > 0.0

    def proximal_loss(self) -> Tensor:
        if not self.enabled:
            raise ValueError("FedProx proximal loss is not enabled.")
        if self.reference_snapshot is None:
            raise ValueError("FedProx reference snapshot is missing.")
        return compute_scaled_fedprox_loss(
            trainable_parameters=self.trainable_parameters,
            reference_snapshot=self.reference_snapshot,
            proximal_mu=self.proximal_mu,
        )

    def add_to_loss(self, loss: Tensor) -> Tensor:
        if not self.enabled:
            return loss
        return loss + self.proximal_loss()


def prepare_lora_training_fedprox(
    *,
    proximal_mu: float,
    trainable_parameters: tuple[nn.Parameter, ...],
) -> LoraTrainingFedProxState:
    """기존 LoRA loop와 동일한 FedProx 검증/snapshot 준비를 수행한다."""

    normalized_mu = validate_proximal_mu(proximal_mu)
    return LoraTrainingFedProxState(
        proximal_mu=normalized_mu,
        trainable_parameters=trainable_parameters,
        reference_snapshot=(
            snapshot_trainable_parameters(trainable_parameters)
            if normalized_mu > 0.0
            else None
        ),
    )
