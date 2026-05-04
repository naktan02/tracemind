"""prototype scoring service bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shared.src.contracts.training_contracts import TrainingObjectiveConfig


@dataclass(slots=True)
class PrototypeScoringRuntime:
    """agent ScoringService를 scripts의 prototype index scorer 뒤에 숨긴다."""

    objective_config: TrainingObjectiveConfig
    similarity_name: str = "cosine"
    shared_state: Any | None = None
    _service: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from agent.src.services.inference.scoring_service import ScoringService

        self._service = ScoringService.from_objective_config(
            self.objective_config,
            similarity_name=self.similarity_name,
            shared_state=self.shared_state,
        )

    def score(
        self,
        embedding: Sequence[float],
        prototypes: Mapping[str, tuple[tuple[float, ...], ...]],
    ) -> dict[str, float]:
        return self._service.score(embedding, prototypes)


def score_prototype_mapping(
    *,
    embedding: Sequence[float],
    prototypes: Mapping[str, tuple[list[float], ...]],
) -> dict[str, float]:
    """기본 prototype scorer로 category score를 계산한다."""

    from agent.src.services.inference.scoring_service import ScoringService

    return ScoringService().score(embedding=embedding, prototypes=prototypes)
