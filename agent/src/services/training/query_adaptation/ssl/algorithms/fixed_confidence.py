"""Fixed-confidence selection baseline."""

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
class FixedConfidenceQuerySslAlgorithm:
    """Top1 confidence threshold만 보는 selection baseline."""

    algorithm_name: str = "top1_confidence_only"

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: QuerySslAlgorithmConfig,
    ) -> AcceptanceDecision:
        return build_acceptance_decision(
            evidence=evidence,
            accepted=evidence.top1_score >= config.confidence_threshold,
        )


__all__ = ["FixedConfidenceQuerySslAlgorithm"]
