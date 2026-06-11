"""SimMatch labeled feature/label memory bank."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SimMatchMemoryBankState:
    """checkpoint에 저장할 SimMatch memory bank state."""

    feature_bank: Tensor
    labels_bank: Tensor


class SimMatchMemoryBank:
    """labeled dataset index로 접근하는 feature/label memory bank."""

    def __init__(
        self,
        *,
        bank_size: int,
        feature_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if bank_size <= 0:
            raise ValueError("bank_size must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.bank_size = int(bank_size)
        self.feature_dim = int(feature_dim)
        self.feature_bank = F.normalize(
            torch.randn(
                (self.feature_dim, self.bank_size),
                dtype=dtype,
                device=device,
            ),
            dim=0,
        )
        self.labels_bank = torch.zeros(
            self.bank_size,
            dtype=torch.long,
            device=device,
        )

    @torch.no_grad()
    def update(
        self,
        *,
        features: Tensor,
        labels: Tensor,
        indices: Tensor,
        ema_bank: float,
    ) -> None:
        """labeled feature/label을 dataset index 위치에 기록한다."""

        features = self._validate_features(features).detach()
        labels = labels.detach().long().reshape(-1)
        indices = indices.detach().long().reshape(-1)
        if features.shape[0] != labels.numel() or features.shape[0] != indices.numel():
            raise ValueError("features, labels, and indices batch size must match.")
        if not 0.0 <= float(ema_bank) <= 1.0:
            raise ValueError("ema_bank must be between 0 and 1.")
        if indices.numel() == 0:
            return
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= self.bank_size:
            raise ValueError("SimMatch labeled row index is outside memory bank.")

        features_t = features.to(
            device=self.feature_bank.device,
            dtype=self.feature_bank.dtype,
        ).t()
        old_features = self.feature_bank[:, indices]
        updated_features = F.normalize(
            float(ema_bank) * old_features + (1.0 - float(ema_bank)) * features_t,
            dim=0,
        )
        self.feature_bank[:, indices] = updated_features
        self.labels_bank[indices] = labels.to(self.labels_bank.device)

    def to(self, device: torch.device | str) -> "SimMatchMemoryBank":
        """bank tensor를 target device로 옮기고 self를 반환한다."""

        self.feature_bank = self.feature_bank.to(device)
        self.labels_bank = self.labels_bank.to(device)
        return self

    def export_state(self) -> SimMatchMemoryBankState:
        """resume checkpoint에 넣을 memory bank state를 반환한다."""

        return SimMatchMemoryBankState(
            feature_bank=self.feature_bank.detach().cpu().clone(),
            labels_bank=self.labels_bank.detach().cpu().clone(),
        )

    def load_state(
        self,
        state: SimMatchMemoryBankState | Mapping[str, Any],
    ) -> None:
        """저장된 memory bank state를 현재 bank shape에 복원한다."""

        state_obj = _coerce_memory_bank_state(state)
        if tuple(state_obj.feature_bank.shape) != (
            self.feature_dim,
            self.bank_size,
        ):
            raise ValueError("feature_bank shape does not match SimMatch memory bank.")
        if tuple(state_obj.labels_bank.shape) != (self.bank_size,):
            raise ValueError("labels_bank shape does not match SimMatch memory bank.")
        self.feature_bank = state_obj.feature_bank.to(
            device=self.feature_bank.device,
            dtype=self.feature_bank.dtype,
        )
        self.labels_bank = state_obj.labels_bank.to(
            device=self.labels_bank.device,
            dtype=torch.long,
        )

    def _validate_features(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [batch, feature_dim].")
        return features


def _coerce_memory_bank_state(
    state: SimMatchMemoryBankState | Mapping[str, Any],
) -> SimMatchMemoryBankState:
    if isinstance(state, SimMatchMemoryBankState):
        return state
    feature_bank = state.get("feature_bank")
    labels_bank = state.get("labels_bank")
    if not isinstance(feature_bank, Tensor):
        raise ValueError("SimMatch memory bank state requires feature_bank tensor.")
    if not isinstance(labels_bank, Tensor):
        raise ValueError("SimMatch memory bank state requires labels_bank tensor.")
    return SimMatchMemoryBankState(
        feature_bank=feature_bank,
        labels_bank=labels_bank,
    )
