"""Acceptance policy base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
    """Agent runtime compatibility가 보는 pseudo-label acceptance metadata."""

    policy_name: str
    selection_hook_name: str
    supported_adapter_kinds: tuple[str, ...]
