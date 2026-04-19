"""Margin-threshold selection baseline."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.acceptance_policies.base import AcceptanceDecision
from agent.src.services.training.acceptance_policies.helpers import (
    build_acceptance_decision,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from ..base import QuerySslAlgorithmConfig


@dataclass(slots=True)
class MarginThresholdQuerySslAlgorithm:
    """Top1 confidence와 top1-top2 margin을 함께 보는 baseline."""

    algorithm_name: str = "top1_margin_threshold"

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: QuerySslAlgorithmConfig,
    ) -> AcceptanceDecision:
        return build_acceptance_decision(
            evidence=evidence,
            accepted=(
                evidence.top1_score >= config.confidence_threshold
                and evidence.margin >= config.margin_threshold
            ),
        )


__all__ = ["MarginThresholdQuerySslAlgorithm"]
