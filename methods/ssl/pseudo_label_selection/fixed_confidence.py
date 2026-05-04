"""Fixed-confidence pseudo-label selection method."""

from __future__ import annotations

from dataclasses import dataclass

from methods.ssl.pseudo_label_selection.base import (
    PseudoLabelSelectionConfig,
    PseudoLabelSelectionDecision,
)
from methods.ssl.pseudo_label_selection.common import (
    build_pseudo_label_selection_decision,
)
from methods.ssl.pseudo_label_selection.registry import (
    register_pseudo_label_selection_method,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@register_pseudo_label_selection_method("top1_confidence_only")
@dataclass(slots=True)
class FixedConfidencePseudoLabelSelectionMethod:
    """Top1 confidence threshold만 보는 selection method."""

    method_name: str = "top1_confidence_only"

    @property
    def hook_name(self) -> str:
        return self.method_name

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> PseudoLabelSelectionDecision:
        return build_pseudo_label_selection_decision(
            evidence=evidence,
            accepted=evidence.top1_score >= config.confidence_threshold,
        )


FixedConfidencePseudoLabelSelectionHook = FixedConfidencePseudoLabelSelectionMethod
