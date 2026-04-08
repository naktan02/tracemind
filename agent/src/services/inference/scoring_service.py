"""프로토타입 점수 계산 서비스."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agent.src.services.inference.scoring_backends import (
    PrototypeSimilarityScoringBackend,
    ScoringBackend,
    build_scoring_backend,
)
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


@dataclass(slots=True)
class ScoringService:
    """설정된 scoring backend로 category score dict를 계산한다."""

    backend: ScoringBackend = field(default_factory=PrototypeSimilarityScoringBackend)

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig,
        *,
        similarity_name: str = "cosine",
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
        return cls(backend=backend)

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    ) -> dict[str, float]:
        return self.backend.score(embedding, prototypes)
