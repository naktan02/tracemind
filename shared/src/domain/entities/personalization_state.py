"""로컬 개인화 상태."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class PersonalizationState:
    """개인 baseline, threshold, prototype 참조를 담는 로컬 상태."""

    schema_version: str
    state_version: str
    baseline_by_category: dict[str, float] = field(default_factory=dict)
    threshold_by_category: dict[str, float] = field(default_factory=dict)
    warmup_status: str = "cold_start"
    updated_at: datetime | None = None
    personal_prototype_refs: dict[str, str] = field(default_factory=dict)
    persistence_features: dict[str, float] = field(default_factory=dict)
    calibration_notes: str | None = None
