"""Federated simulation이 server round draft로 넘길 task config surface."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.training_contracts import (
    SecureAggregationConfig,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


class FederatedTrainingTaskConfig(Protocol):
    """simulation request와 server mapper 사이의 최소 task config port."""

    task_type: TrainingTaskType
    local_epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    objective_config: (
        TrainingObjectiveConfig | Mapping[str, TrainingConfigScalar] | None
    )
    selection_policy: (
        TrainingSelectionPolicy | Mapping[str, TrainingConfigScalar] | None
    )
    secure_aggregation: (
        SecureAggregationConfig | Mapping[str, TrainingConfigScalar] | bool | None
    )
    min_required_examples: int | None
    gradient_clip_norm: float | None
    deadline_at: datetime | None
    notes: str | None
