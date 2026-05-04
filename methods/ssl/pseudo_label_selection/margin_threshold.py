"""Margin-threshold pseudo-label selection method."""

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


@register_pseudo_label_selection_method("top1_margin_threshold")
@dataclass(slots=True)
class MarginThresholdPseudoLabelSelectionMethod:
    """Top1 confidence와 top1-top2 margin을 함께 보는 selection method."""

    method_name: str = "top1_margin_threshold"

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
            accepted=(
                evidence.top1_score >= config.confidence_threshold
                and evidence.margin >= config.margin_threshold
            ),
        )


MarginThresholdPseudoLabelSelectionHook = MarginThresholdPseudoLabelSelectionMethod
