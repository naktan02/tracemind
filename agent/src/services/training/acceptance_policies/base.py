"""Acceptance policy base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(slots=True)
class AcceptanceDecision:
    """Pseudo-label acceptance 판단 결과."""

    label: str
    confidence: float
    confidence_kind: str | None
    margin: float
    accepted: bool
    runner_up_label: str | None = None
    runner_up_score: float | None = None
    sample_weight: float = 1.0


class PseudoLabelAcceptancePolicy(Protocol):
    """Evidence를 pseudo-label 후보로 해석하는 정책."""

    policy_name: str
    supported_adapter_kinds: tuple[str, ...]

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision:
        """Evidence를 해석해 acceptance 결과를 만든다."""
