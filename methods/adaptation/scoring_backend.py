"""Method-owned scoring backend contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from shared.src.contracts.scoring_contracts import ScoringConfigPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

ScoringAssets = Mapping[str, Sequence[float] | Sequence[Sequence[float]]]


class SharedAdapterScoringBackend(Protocol):
    """category score dict를 계산하는 method-owned backend interface."""

    backend_name: str
    supported_adapter_kinds: tuple[str, ...]
    requires_shared_state: bool

    def score(
        self,
        embedding: Sequence[float],
        scoring_assets: ScoringAssets,
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        """임베딩과 optional shared state로 category score dict를 계산한다."""


SharedAdapterScoringBackendFactory = Callable[
    [ScoringConfigPayload, str],
    SharedAdapterScoringBackend,
]
