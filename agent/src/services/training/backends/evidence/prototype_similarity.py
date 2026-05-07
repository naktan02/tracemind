"""Prototype similarity evidence backend."""

from __future__ import annotations

from dataclasses import dataclass

from methods.prototype.evidence.helpers import (
    build_ranked_evidence,
    rank_category_scores,
)
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import ANY_ADAPTER_KIND, PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME
from .registry import register_pseudo_label_evidence_backend

PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    display_name=PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    implementation_module=(
        "agent.src.services.training.backends.evidence.prototype_similarity"
    ),
    core_method_name=PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    family_name="pseudo_label_evidence",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)


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


@register_pseudo_label_evidence_backend(
    PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    catalog_entry=PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_CATALOG_ENTRY,
)
def build_prototype_similarity_evidence_backend(
    objective_config: TrainingObjectiveConfig,
) -> PrototypeSimilarityEvidenceBackend:
    """registry용 prototype-similarity evidence backend factory."""

    del objective_config
    return PrototypeSimilarityEvidenceBackend()
