"""로컬 판단 결과."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AssessmentResult:
    """개인 신호와 또래 신호를 결합한 최종 로컬 평가 결과."""

    schema_version: str
    assessment_id: str
    decision: str
    explanation: str
    self_shift: float | None = None
    peer_deviation: float | None = None
    persistence: float | None = None
