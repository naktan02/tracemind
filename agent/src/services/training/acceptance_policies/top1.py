"""Top1-based acceptance policies."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.ssl.hooks.pseudo_label_selection import (
    registry as pseudo_label_selection_registry,
)
from agent.src.services.training.ssl.hooks.pseudo_label_selection.base import (
    PseudoLabelSelectionConfig,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .base import AcceptanceDecision


@dataclass(slots=True)
class Top1MarginThresholdAcceptancePolicy:
    """Top1 confidence와 top1-top2 margin을 함께 본다."""

    policy_name: str = "top1_margin_threshold"
    supported_adapter_kinds: tuple[str, ...] = ("*",)

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        return pseudo_label_selection_registry.build_pseudo_label_selection_hook(
            self.policy_name
        ).evaluate(
            evidence=evidence,
            config=PseudoLabelSelectionConfig(
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            ),
        )


@dataclass(slots=True)
class Top1ConfidenceOnlyAcceptancePolicy:
    """Top1 confidence만으로 pseudo-label 채택 여부를 결정한다."""

    policy_name: str = "top1_confidence_only"
    supported_adapter_kinds: tuple[str, ...] = ("*",)

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        return pseudo_label_selection_registry.build_pseudo_label_selection_hook(
            self.policy_name
        ).evaluate(
            evidence=evidence,
            config=PseudoLabelSelectionConfig(
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            ),
        )
