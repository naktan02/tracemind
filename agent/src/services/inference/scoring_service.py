"""로컬 category score 계산 서비스."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent.src.services.inference.scoring_backends.base import (
    ScoringAssets,
    ScoringBackend,
)
from agent.src.services.inference.scoring_backends.helpers import (
    resolve_scoring_backend_name,
)
from agent.src.services.inference.scoring_backends.registry import (
    build_scoring_backend,
)
from shared.src.contracts.scoring_contracts import ScoringConfigPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


@dataclass(slots=True)
class ScoringService:
    """설정된 scoring backend로 category score dict를 계산한다."""

    backend: ScoringBackend
    shared_state: SharedAdapterState | None = None

    @classmethod
    def from_scoring_config(
        cls,
        scoring_config: ScoringConfigPayload,
        *,
        similarity_name: str = "cosine",
        shared_state: SharedAdapterState | None = None,
    ) -> "ScoringService":
        """Scoring config로부터 scoring service를 조립한다."""

        backend_name = scoring_config.scorer_backend_name
        backend = build_scoring_backend(
            backend_name,
            scoring_config=scoring_config,
            similarity_name=similarity_name,
        )
        return cls(backend=backend, shared_state=shared_state)

    def score(
        self,
        embedding: Sequence[float],
        scoring_assets: ScoringAssets,
        *,
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        effective_shared_state = (
            shared_state if shared_state is not None else self.shared_state
        )
        return self.backend.score(
            embedding,
            scoring_assets,
            shared_state=effective_shared_state,
        )

    @property
    def backend_name(self) -> str:
        """현재 backend canonical name."""

        return resolve_scoring_backend_name(self.backend)
