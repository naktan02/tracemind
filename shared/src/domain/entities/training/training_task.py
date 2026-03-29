"""학습 작업 정의."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class TrainingTask:
    """중앙이 로컬에 배포하는 학습 작업 정의."""

    schema_version: str
    task_id: str
    round_id: str
    model_id: str
    model_revision: str
    task_type: str
    training_scope: str
    local_epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    objective_config: dict[str, str | int | float | bool] = field(default_factory=dict)
    selection_policy: dict[str, str | int | float | bool] = field(default_factory=dict)
    deadline_at: datetime | None = None
    gradient_clip_norm: float | None = None
    min_required_examples: int | None = None
    secure_aggregation_required: bool = False
    notes: str | None = None
