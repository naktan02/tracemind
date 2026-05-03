"""Fixed-confidence pseudo-label selection hook."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.acceptance_policies.base import AcceptanceDecision
from agent.src.services.training.acceptance_policies.helpers import (
    build_acceptance_decision,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import PseudoLabelSelectionConfig
from .registry import register_pseudo_label_selection_hook


@register_pseudo_label_selection_hook("top1_confidence_only")
@dataclass(slots=True)
class FixedConfidencePseudoLabelSelectionHook:
    """Top1 confidence threshold만 보는 selection hook."""

    hook_name: str = "top1_confidence_only"

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> AcceptanceDecision:
        return build_acceptance_decision(
            evidence=evidence,
            accepted=evidence.top1_score >= config.confidence_threshold,
        )
