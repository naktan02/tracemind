"""로컬 학습에 쓰는 피드백 신호."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class DecisionFeedbackSignal:
    """pseudo-label 또는 사용자/시스템 feedback을 표현하는 신호."""

    schema_version: str
    signal_id: str
    signal_type: str
    label: str
    confidence: float
    occurred_at: datetime
    source_event_ref: str | None = None
    task_context: dict[str, str | int | float | bool] = field(default_factory=dict)
    notes: str | None = None
