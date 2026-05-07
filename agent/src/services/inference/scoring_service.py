"""프로토타입 점수 계산 서비스."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agent.src.services.inference.scoring_backends.base import ScoringBackend
from agent.src.services.inference.scoring_backends.helpers import (
    resolve_scoring_backend_name,
    resolve_scoring_confidence_kind,
)
from agent.src.services.inference.scoring_backends.prototype_similarity import (
    PrototypeSimilarityScoringBackend,
)
from agent.src.services.inference.scoring_backends.registry import (
    build_scoring_backend,
)
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


@dataclass(slots=True)
class ScoringService:
    """설정된 scoring backend로 category score dict를 계산한다."""

    backend: ScoringBackend = field(default_factory=PrototypeSimilarityScoringBackend)
    shared_state: SharedAdapterState | None = None

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig,
        *,
        similarity_name: str = "cosine",
        shared_state: SharedAdapterState | None = None,
    ) -> "ScoringService":
        """학습 objective config로부터 scoring service를 조립한다."""

        backend_name = (
            objective_config.scorer_backend_name
            or DEFAULT_TRAINING_PROFILE.scorer_backend_name
        )
        backend = build_scoring_backend(
            backend_name,
            objective_config=objective_config,
            similarity_name=similarity_name,
        )
        return cls(backend=backend, shared_state=shared_state)

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
        *,
        shared_state: SharedAdapterState | None = None,
    ) -> dict[str, float]:
        effective_shared_state = (
            shared_state if shared_state is not None else self.shared_state
        )
        return self.backend.score(
            embedding,
            prototypes,
            shared_state=effective_shared_state,
        )

    @property
    def backend_name(self) -> str:
        """현재 backend canonical name."""

        return resolve_scoring_backend_name(self.backend)

    @property
    def confidence_kind(self) -> str:
        """query buffer에 남길 confidence kind."""

        return resolve_scoring_confidence_kind(self.backend)
