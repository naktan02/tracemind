"""CoMatch류 feature/probability memory bank primitive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CoMatchMemoryBankState:
    """checkpoint에 저장할 CoMatch memory bank state."""

    feature_queue: Tensor
    probability_queue: Tensor
    queue_ptr: int
    filled_size: int


class CoMatchMemoryBank:
    """weak feature/probability queue와 memory smoothing을 소유한다."""

    def __init__(
        self,
        *,
        queue_size: int,
        feature_dim: int,
        num_classes: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.queue_size = int(queue_size)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.feature_queue = torch.zeros(
            (self.queue_size, self.feature_dim),
            dtype=dtype,
            device=device,
        )
        self.probability_queue = torch.zeros(
            (self.queue_size, self.num_classes),
            dtype=dtype,
            device=device,
        )
        self.queue_ptr = 0
        self.filled_size = 0

    @torch.no_grad()
    def update(
        self,
        *,
        features: Tensor,
        probabilities: Tensor,
    ) -> None:
        """features/probabilities batch를 ring buffer에 기록한다."""

        features = self._validate_features(features).detach()
        probabilities = self._validate_probabilities(probabilities).detach()
        if features.shape[0] != probabilities.shape[0]:
            raise ValueError("features and probabilities batch size must match.")
        if features.shape[0] == 0:
            return
        features = features.to(
            device=self.feature_queue.device,
            dtype=self.feature_queue.dtype,
        )
        probabilities = probabilities.to(
            device=self.probability_queue.device,
            dtype=self.probability_queue.dtype,
        )

        remaining_features = features
        remaining_probabilities = probabilities
        while remaining_features.shape[0] > 0:
            write_count = min(
                remaining_features.shape[0],
                self.queue_size - self.queue_ptr,
            )
            write_end = self.queue_ptr + write_count
            self.feature_queue[self.queue_ptr : write_end] = remaining_features[
                :write_count
            ]
            self.probability_queue[self.queue_ptr : write_end] = (
                remaining_probabilities[:write_count]
            )
            self.queue_ptr = write_end % self.queue_size
            self.filled_size = min(self.queue_size, self.filled_size + write_count)
            remaining_features = remaining_features[write_count:]
            remaining_probabilities = remaining_probabilities[write_count:]

    def smooth_probabilities(
        self,
        *,
        features: Tensor,
        probabilities: Tensor,
        temperature: float,
        smoothing_alpha: float,
    ) -> Tensor:
        """memory feature similarity로 probability를 smoothing한다."""

        if self.filled_size <= 0:
            return probabilities
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be between 0 and 1.")
        features = self._validate_features(features).to(
            device=self.feature_queue.device,
            dtype=self.feature_queue.dtype,
        )
        probabilities = self._validate_probabilities(probabilities).to(
            device=self.probability_queue.device,
            dtype=self.probability_queue.dtype,
        )
        active_features = self.active_features()
        active_probabilities = self.active_probabilities()
        affinities = torch.exp(features @ active_features.t() / float(temperature))
        affinities = affinities / affinities.sum(dim=1, keepdim=True).clamp(min=1e-12)
        memory_probabilities = affinities @ active_probabilities
        return (
            smoothing_alpha * probabilities
            + (1.0 - smoothing_alpha) * memory_probabilities
        )

    def active_features(self) -> Tensor:
        """현재 채워진 feature queue prefix를 반환한다."""

        return self.feature_queue[: self.filled_size]

    def active_probabilities(self) -> Tensor:
        """현재 채워진 probability queue prefix를 반환한다."""

        return self.probability_queue[: self.filled_size]

    def export_state(self) -> CoMatchMemoryBankState:
        """resume checkpoint에 넣을 memory bank state를 반환한다."""

        return CoMatchMemoryBankState(
            feature_queue=self.feature_queue.detach().cpu().clone(),
            probability_queue=self.probability_queue.detach().cpu().clone(),
            queue_ptr=int(self.queue_ptr),
            filled_size=int(self.filled_size),
        )

    def load_state(
        self,
        state: CoMatchMemoryBankState | Mapping[str, Any],
    ) -> None:
        """저장된 memory bank state를 현재 queue shape에 복원한다."""

        state_obj = _coerce_memory_bank_state(state)
        if tuple(state_obj.feature_queue.shape) != (
            self.queue_size,
            self.feature_dim,
        ):
            raise ValueError("feature_queue shape does not match memory bank.")
        if tuple(state_obj.probability_queue.shape) != (
            self.queue_size,
            self.num_classes,
        ):
            raise ValueError("probability_queue shape does not match memory bank.")
        if not 0 <= int(state_obj.queue_ptr) < self.queue_size:
            raise ValueError("queue_ptr must be inside queue_size.")
        if not 0 <= int(state_obj.filled_size) <= self.queue_size:
            raise ValueError("filled_size must be between 0 and queue_size.")
        self.feature_queue = state_obj.feature_queue.to(
            device=self.feature_queue.device,
            dtype=self.feature_queue.dtype,
        )
        self.probability_queue = state_obj.probability_queue.to(
            device=self.probability_queue.device,
            dtype=self.probability_queue.dtype,
        )
        self.queue_ptr = int(state_obj.queue_ptr)
        self.filled_size = int(state_obj.filled_size)

    def _validate_features(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [batch, feature_dim].")
        return features

    def _validate_probabilities(self, probabilities: Tensor) -> Tensor:
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError("probabilities must have shape [batch, num_classes].")
        return probabilities


def _coerce_memory_bank_state(
    state: CoMatchMemoryBankState | Mapping[str, Any],
) -> CoMatchMemoryBankState:
    if isinstance(state, CoMatchMemoryBankState):
        return state
    feature_queue = state.get("feature_queue")
    probability_queue = state.get("probability_queue")
    if not isinstance(feature_queue, Tensor):
        raise ValueError("memory bank state requires feature_queue tensor.")
    if not isinstance(probability_queue, Tensor):
        raise ValueError("memory bank state requires probability_queue tensor.")
    return CoMatchMemoryBankState(
        feature_queue=feature_queue,
        probability_queue=probability_queue,
        queue_ptr=int(state.get("queue_ptr", 0)),
        filled_size=int(state.get("filled_size", feature_queue.shape[0])),
    )
