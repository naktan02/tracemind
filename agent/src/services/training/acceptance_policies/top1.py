"""Top1-based acceptance policies."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.query_adaptation.ssl.base import (
    QuerySslAlgorithmConfig,
)
from agent.src.services.training.query_adaptation.ssl.registry import (
    build_query_ssl_algorithm,
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
        return build_query_ssl_algorithm(self.policy_name).evaluate(
            evidence=evidence,
            config=QuerySslAlgorithmConfig(
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
        return build_query_ssl_algorithm(self.policy_name).evaluate(
            evidence=evidence,
            config=QuerySslAlgorithmConfig(
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            ),
        )


__all__ = [
    "Top1ConfidenceOnlyAcceptancePolicy",
    "Top1MarginThresholdAcceptancePolicy",
]
