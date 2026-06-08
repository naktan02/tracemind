"""Pseudo-label evidence 조립 서비스."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.backends.evidence.base import (
    ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME,
    PseudoLabelEvidenceBackend,
)
from agent.src.services.training.backends.evidence.registry import (
    build_pseudo_label_evidence_backend,
)
from agent.src.services.training.selection.stored_event_defaults import (
    STORED_EVENT_EVIDENCE_BACKEND_NAME,
)
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(slots=True)
class PseudoLabelEvidenceService:
    """TrainingTask와 AnalysisEvent를 evidence 계층으로 연결한다."""

    default_evidence_backend_name: str = STORED_EVENT_EVIDENCE_BACKEND_NAME

    def __post_init__(self) -> None:
        if self.default_evidence_backend_name != ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME:
            raise ValueError(
                "Default pseudo-label evidence backend does not match the "
                "configured default objective profile."
            )

    def build_evidences(
        self,
        *,
        analysis_events: tuple[AnalysisEvent, ...] | list[AnalysisEvent],
        training_task: TrainingTask,
    ) -> tuple[PseudoLabelEvidence, ...]:
        backend = self._resolve_backend(training_task=training_task)
        return tuple(
            backend.build_evidence(analysis_event=analysis_event)
            for analysis_event in analysis_events
        )

    def _resolve_backend(
        self,
        *,
        training_task: TrainingTask,
    ) -> PseudoLabelEvidenceBackend:
        backend_name = (
            training_task.objective_config.evidence_backend_name
            or self.default_evidence_backend_name
        )
        return build_pseudo_label_evidence_backend(
            backend_name,
            objective_config=training_task.objective_config,
        )
