"""Pseudo-label evidence 조립 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.services.training.backends.evidence.base import (
    PseudoLabelEvidenceBackend,
)
from agent.src.services.training.backends.evidence.prototype_similarity import (
    PrototypeSimilarityEvidenceBackend,
)
from agent.src.services.training.backends.evidence.registry import (
    build_pseudo_label_evidence_backend,
)
from shared.src.config.training_defaults import (
    DEFAULT_TRAINING_PROFILE,
    TrainingDefaultsProfile,
)
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(slots=True)
class PseudoLabelEvidenceService:
    """TrainingTask와 ScoredEvent를 evidence 계층으로 연결한다."""

    default_profile: TrainingDefaultsProfile = field(
        default=DEFAULT_TRAINING_PROFILE
    )
    default_backend: PseudoLabelEvidenceBackend = field(
        default_factory=PrototypeSimilarityEvidenceBackend
    )

    def __post_init__(self) -> None:
        if self.default_evidence_backend_name != self.default_backend.backend_name:
            raise ValueError(
                "Default pseudo-label evidence backend does not match the "
                "configured default objective profile."
            )

    @property
    def default_evidence_backend_name(self) -> str:
        return self.default_profile.evidence_backend_name

    def build_evidences(
        self,
        *,
        scored_events: tuple[ScoredEvent, ...] | list[ScoredEvent],
        training_task: TrainingTask,
    ) -> tuple[PseudoLabelEvidence, ...]:
        backend = self._resolve_backend(training_task=training_task)
        return tuple(
            backend.build_evidence(scored_event=scored_event)
            for scored_event in scored_events
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
        if backend_name == self.default_backend.backend_name:
            return self.default_backend
        return build_pseudo_label_evidence_backend(
            backend_name,
            objective_config=training_task.objective_config,
        )


__all__ = ["PseudoLabelEvidenceService"]
