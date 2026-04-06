"""로컬 학습 업데이트 메타데이터."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class TrainingUpdateEnvelope:
    """로컬 agent가 중앙으로 보내는 업데이트 envelope."""

    schema_version: str
    update_id: str
    round_id: str
    task_id: str
    model_id: str
    base_model_revision: str
    training_scope: str
    payload_ref: str
    payload_format: str
    example_count: int
    client_metrics: dict[str, float] = field(default_factory=dict)
    created_at: datetime | None = None
    clipped: bool | None = None
    dp_applied: bool | None = None
    checksum: str | None = None
    agent_id: str | None = None
    notes: str | None = None
