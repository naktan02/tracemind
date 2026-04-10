"""FixMatch-style weak-view evidence backend."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import ANY_ADAPTER_KIND, FIXMATCH_WEAK_VIEW_EVIDENCE_BACKEND_NAME
from .helpers import build_ranked_evidence, rank_category_scores, softmax_distribution

FIXMATCH_EVIDENCE_EXTRA_SCOPE = "evidence_backend"
FIXMATCH_EVIDENCE_LEGACY_KEYS = ("fixmatch.temperature",)


@dataclass(slots=True)
class FixMatchWeakViewEvidenceBackend:
    """weak-view score를 posterior-like evidence로 정규화한다."""

    temperature: float = 1.0
    backend_name: str = FIXMATCH_WEAK_VIEW_EVIDENCE_BACKEND_NAME
    supported_adapter_kinds: tuple[str, ...] = (ANY_ADAPTER_KIND,)
    confidence_kind: str = "posterior_probability"
    view_kind: str = "weak_view"

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig,
    ) -> "FixMatchWeakViewEvidenceBackend":
        extras = objective_config.get_component_extras(
            FIXMATCH_EVIDENCE_EXTRA_SCOPE,
            legacy_keys=FIXMATCH_EVIDENCE_LEGACY_KEYS,
        )
        temperature_raw = extras.get("temperature", 1.0)
        return cls(temperature=float(temperature_raw))

    def build_evidence(
        self,
        *,
        scored_event: ScoredEvent,
    ) -> PseudoLabelEvidence:
        distribution = softmax_distribution(
            scored_event.category_scores,
            temperature=self.temperature,
        )
        ranked_scores = rank_category_scores(distribution)
        return build_ranked_evidence(
            scored_event=scored_event,
            ranked_scores=ranked_scores,
            confidence_kind=self.confidence_kind,
            view_kind=self.view_kind,
            backend_name=self.backend_name,
            label_distribution=distribution,
            sample_weight=1.0,
            metadata={"temperature": self.temperature},
        )


__all__ = [
    "FIXMATCH_EVIDENCE_EXTRA_SCOPE",
    "FixMatchWeakViewEvidenceBackend",
]
