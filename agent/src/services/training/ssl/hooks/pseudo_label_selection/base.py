"""Pseudo-label selection hook base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent.src.services.training.acceptance_policies.base import AcceptanceDecision
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionConfig:
    """Selection hook이 해석할 threshold 입력."""

    confidence_threshold: float
    margin_threshold: float = 0.0


class PseudoLabelSelectionHook(Protocol):
    """중앙/FL SSL에서 재사용하는 pseudo-label selection hook."""

    hook_name: str

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> AcceptanceDecision:
        """Evidence 하나를 selection decision 하나로 해석한다."""
