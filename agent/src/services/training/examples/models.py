"""로컬 학습 example 공용 모델."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


@dataclass(slots=True)
class EmbeddedTrainingExample:
    """학습 후보가 된 로컬 analysis event와 임베딩."""

    analysis_event: AnalysisEvent
    embedding: list[float]
    base_embedding: list[float] | None = None
    candidate: PseudoLabelCandidate | None = None
    view_kind: str = "single_view"
    weak_analysis_event: AnalysisEvent | None = None
    weak_embedding: list[float] | None = None
    strong_analysis_event: AnalysisEvent | None = None
    strong_embedding: list[float] | None = None
    strong_base_embedding: list[float] | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def evidence_analysis_event(self) -> AnalysisEvent:
        """Pseudo-label evidence를 계산할 기준 view."""

        return self.weak_analysis_event or self.analysis_event

    @property
    def update_analysis_event(self) -> AnalysisEvent:
        """학습/update를 적용할 기준 view."""

        return self.strong_analysis_event or self.analysis_event

    @property
    def update_embedding(self) -> list[float]:
        """학습/update backend가 사용할 대표 임베딩."""

        return self.strong_embedding or self.embedding

    @property
    def selection_key(self) -> str:
        """selection 결과와 training example 재매칭에 쓰는 key."""

        return self.evidence_analysis_event.query_id
