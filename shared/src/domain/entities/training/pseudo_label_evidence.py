"""공통 pseudo-label evidence 표현."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

PSEUDO_LABEL_EVIDENCE_V1 = "pseudo_label_evidence.v1"


@dataclass(slots=True)
class PseudoLabelEvidence:
    """방법별 신호를 공통 pseudo-label 판단 입력으로 정규화한 표현."""

    schema_version: str
    evidence_id: str
    source_event_ref: str
    occurred_at: datetime
    label: str
    confidence: float
    confidence_kind: str
    margin: float
    top1_label: str
    top1_score: float
    top2_label: str | None = None
    top2_score: float = 0.0
    sample_weight: float = 1.0
    view_kind: str = "single_view"
    raw_scores: dict[str, float] = field(default_factory=dict)
    label_distribution: dict[str, float] | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
