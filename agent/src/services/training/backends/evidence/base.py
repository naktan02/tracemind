"""Evidence backend base types."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

ANY_ADAPTER_KIND = "*"
PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME = "prototype_similarity_evidence"


class PseudoLabelEvidenceBackend(Protocol):
    """ScoredEvent를 공통 pseudo-label evidence로 정규화한다."""

    backend_name: str
    supported_adapter_kinds: tuple[str, ...]

    def build_evidence(
        self,
        *,
        scored_event: ScoredEvent,
    ) -> PseudoLabelEvidence:
        """ScoredEvent 하나를 evidence 하나로 변환한다."""


PseudoLabelEvidenceBackendFactory = Callable[
    [TrainingObjectiveConfig],
    PseudoLabelEvidenceBackend,
]


__all__ = [
    "ANY_ADAPTER_KIND",
    "PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME",
    "PseudoLabelEvidenceBackend",
    "PseudoLabelEvidenceBackendFactory",
]
