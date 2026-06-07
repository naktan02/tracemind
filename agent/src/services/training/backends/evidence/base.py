"""Evidence backend base types."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

ANY_ADAPTER_KIND = "*"
ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME = "analysis_score_evidence"
PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME = "prototype_similarity_evidence"


class PseudoLabelEvidenceBackend(Protocol):
    """AnalysisEvent를 공통 pseudo-label evidence로 정규화한다."""

    backend_name: str
    supported_adapter_kinds: tuple[str, ...]

    def build_evidence(
        self,
        *,
        analysis_event: AnalysisEvent,
    ) -> PseudoLabelEvidence:
        """AnalysisEvent 하나를 evidence 하나로 변환한다."""


PseudoLabelEvidenceBackendFactory = Callable[
    [TrainingObjectiveConfig],
    PseudoLabelEvidenceBackend,
]
