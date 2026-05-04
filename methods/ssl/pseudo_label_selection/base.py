"""Pseudo-label selection method base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionConfig:
    """Selection method가 해석할 threshold 입력."""

    confidence_threshold: float
    margin_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionDecision:
    """순수 selection rule의 판단 결과."""

    label: str
    confidence: float
    confidence_kind: str | None
    margin: float
    accepted: bool
    runner_up_label: str | None = None
    runner_up_score: float | None = None
    sample_weight: float = 1.0


class PseudoLabelSelectionMethod(Protocol):
    """중앙/FL SSL에서 재사용하는 pseudo-label selection method."""

    method_name: str

    @property
    def hook_name(self) -> str:
        """기존 agent runtime 용어와 맞춘 method name alias."""
        ...

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        config: PseudoLabelSelectionConfig,
    ) -> PseudoLabelSelectionDecision:
        """Evidence 하나를 selection decision 하나로 해석한다."""


PseudoLabelSelectionHook = PseudoLabelSelectionMethod
