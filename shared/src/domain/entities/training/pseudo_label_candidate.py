"""로컬 pseudo-label 후보."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class PseudoLabelCandidate:
    """prototype score에서 뽑은 로컬 pseudo-label 후보."""

    schema_version: str
    candidate_id: str
    source_event_ref: str
    occurred_at: datetime
    label: str
    confidence: float
    margin: float
    accepted: bool
    runner_up_label: str | None = None
    runner_up_score: float | None = None
    task_id: str | None = None
    round_id: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
