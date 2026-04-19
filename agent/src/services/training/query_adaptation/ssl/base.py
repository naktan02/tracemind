"""Query adaptation SSL selection algorithm base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent.src.services.training.acceptance_policies.base import AcceptanceDecision
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(frozen=True, slots=True)
class QuerySslAlgorithmConfig:
    """Selection algorithm이 해석할 threshold 입력."""

    confidence_threshold: float
    margin_threshold: float = 0.0


class QuerySslAlgorithm(Protocol):
    """Query adaptation용 SSL selection 알고리즘 인터페이스."""

    algorithm_name: str

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: QuerySslAlgorithmConfig,
    ) -> AcceptanceDecision:
        """Evidence 하나를 selection decision 하나로 해석한다."""


__all__ = ["QuerySslAlgorithm", "QuerySslAlgorithmConfig"]
