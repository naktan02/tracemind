"""Scoring backend base types."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

ScoringAssets = Mapping[str, Sequence[float] | Sequence[Sequence[float]]]


class ScoringBackend(Protocol):
    """category score dict를 계산하는 backend 인터페이스."""

    backend_name: str
    confidence_kind: str
    supported_adapter_kinds: tuple[str, ...]
    requires_shared_state: bool

    def score(
        self,
        embedding: Sequence[float],
        scoring_assets: ScoringAssets,
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        """임베딩과 scorer별 asset으로 category score dict를 계산한다."""


ScoringBackendFactory = Callable[[TrainingObjectiveConfig, str], ScoringBackend]
