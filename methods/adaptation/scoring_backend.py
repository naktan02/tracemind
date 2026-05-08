"""Method-owned scoring backend contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


class SharedAdapterScoringBackend(Protocol):
    """category score dict를 계산하는 method-owned backend interface."""

    backend_name: str
    confidence_kind: str
    supported_adapter_kinds: tuple[str, ...]
    requires_shared_state: bool

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        """임베딩과 optional shared state로 category score dict를 계산한다."""


SharedAdapterScoringBackendFactory = Callable[
    [TrainingObjectiveConfig, str],
    SharedAdapterScoringBackend,
]
