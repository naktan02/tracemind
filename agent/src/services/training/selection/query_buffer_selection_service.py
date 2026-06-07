"""Query buffer 기반 pseudo-label selection runner."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionResult,
    PseudoLabelSelectionService,
)
from agent.src.services.training.selection.query_buffer_projection import (
    build_query_buffer_evidences,
)
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.inference.events import AnalysisEvent


@dataclass(slots=True)
class QueryBufferSelectionService:
    """Query buffer snapshot을 기존 selection 서비스에 연결한다."""

    selector: PseudoLabelSelectionService = field(
        default_factory=PseudoLabelSelectionService
    )

    def select(
        self,
        *,
        records: tuple[QueryBufferRecord, ...] | list[QueryBufferRecord],
        analysis_events: tuple[AnalysisEvent, ...] | list[AnalysisEvent],
        training_task: TrainingTask,
    ) -> PseudoLabelSelectionResult:
        evidences = build_query_buffer_evidences(
            records=records,
            analysis_events=analysis_events,
        )
        return self.selector.select_evidences(
            evidences=evidences,
            training_task=training_task,
        )
