"""로컬 학습 example 공용 모델."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


@dataclass(slots=True)
class EmbeddedTrainingExample:
    """학습 후보가 된 로컬 scored event와 임베딩."""

    scored_event: ScoredEvent
    embedding: list[float]
    base_embedding: list[float] | None = None
    candidate: PseudoLabelCandidate | None = None


__all__ = ["EmbeddedTrainingExample"]
