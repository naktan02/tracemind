"""Method-agnostic analysis score evidence backend."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.selection.evidence_projection import (
    build_ranked_evidence,
    rank_category_scores,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME, ANY_ADAPTER_KIND
from .registry import register_pseudo_label_evidence_backend

ANALYSIS_SCORE_EVIDENCE_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME,
    display_name=ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME,
    implementation_module="agent.src.services.training.backends.evidence.analysis_score",
    core_method_name=ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME,
    family_name="pseudo_label_evidence",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)


@dataclass(slots=True)
class AnalysisScoreEvidenceBackend:
    """AnalysisEvent의 category score를 공통 evidence로 정규화한다."""

    backend_name: str = ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME
    supported_adapter_kinds: tuple[str, ...] = (ANY_ADAPTER_KIND,)
    view_kind: str = "single_view"

    def build_evidence(
        self,
        *,
        analysis_event: AnalysisEvent,
    ) -> PseudoLabelEvidence:
        ranked_scores = rank_category_scores(analysis_event.category_scores)
        return build_ranked_evidence(
            analysis_event=analysis_event,
            ranked_scores=ranked_scores,
            view_kind=self.view_kind,
            backend_name=self.backend_name,
        )


@register_pseudo_label_evidence_backend(
    ANALYSIS_SCORE_EVIDENCE_BACKEND_NAME,
    catalog_entry=ANALYSIS_SCORE_EVIDENCE_BACKEND_CATALOG_ENTRY,
)
def build_analysis_score_evidence_backend(
    objective_config: TrainingObjectiveConfig,
) -> AnalysisScoreEvidenceBackend:
    """registry용 method-agnostic score evidence backend factory."""

    del objective_config
    return AnalysisScoreEvidenceBackend()
