"""로컬 판단 결과 entity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AssessmentResult:
    """로컬 시계열 rule 판단 결과."""

    schema_version: str
    assessment_id: str
    decision: str
    explanation: str
    global_score: float | None = None
    baseline_shift: float | None = None
    persistence: float | None = None
    confidence: float | None = None
    focus_category: str | None = None
