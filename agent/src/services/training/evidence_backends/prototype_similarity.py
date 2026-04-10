"""Prototype similarity evidence backend."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import ANY_ADAPTER_KIND, PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME
from .helpers import build_ranked_evidence, rank_category_scores


@dataclass(slots=True)
class PrototypeSimilarityEvidenceBackend:
    """현재 prototype similarity score를 공통 evidence로 정규화한다."""

    backend_name: str = PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME
    supported_adapter_kinds: tuple[str, ...] = (ANY_ADAPTER_KIND,)
    confidence_kind: str = "prototype_similarity"
    view_kind: str = "single_view"

    def build_evidence(
        self,
        *,
        scored_event: ScoredEvent,
    ) -> PseudoLabelEvidence:
        ranked_scores = rank_category_scores(scored_event.category_scores)
        return build_ranked_evidence(
            scored_event=scored_event,
            ranked_scores=ranked_scores,
            confidence_kind=self.confidence_kind,
            view_kind=self.view_kind,
            backend_name=self.backend_name,
        )


__all__ = ["PrototypeSimilarityEvidenceBackend"]
