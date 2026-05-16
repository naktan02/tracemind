"""Round request mapping helpers for federated simulation."""

from __future__ import annotations

from collections.abc import Mapping

from main_server.src.services.federation.rounds.boundary.models import (
    RoundOpenDraftRequest,
    RoundTaskConfig,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.contracts.training_contracts import (
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)

FederatedRoundOpenRequest = RoundOpenDraftRequest


def build_federated_training_task_config(
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    min_required_examples: int,
    gradient_clip_norm: float | None,
    objective_config: (
        TrainingObjectiveConfig | Mapping[str, TrainingConfigScalar] | None
    ),
    selection_policy: (
        TrainingSelectionPolicy | Mapping[str, TrainingConfigScalar] | None
    ),
) -> FederatedTrainingTaskConfig:
    """FL simulation용 RoundTaskConfig 생성을 main_server bridge로 격리한다."""

    return RoundTaskConfig(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        min_required_examples=min_required_examples,
        gradient_clip_norm=gradient_clip_norm,
        objective_config=objective_config,
        selection_policy=selection_policy,
    )


def build_round_open_request(
    *,
    round_id: str,
    training_task_config: FederatedTrainingTaskConfig,
) -> FederatedRoundOpenRequest:
    """simulation task template을 canonical round open request로 변환한다."""

    return RoundOpenDraftRequest(
        round_id=round_id,
        task_type=training_task_config.task_type,
        local_epochs=training_task_config.local_epochs,
        batch_size=training_task_config.batch_size,
        learning_rate=training_task_config.learning_rate,
        max_steps=training_task_config.max_steps,
        objective_config=training_task_config.objective_config,
        selection_policy=training_task_config.selection_policy,
        secure_aggregation=training_task_config.secure_aggregation,
        min_required_examples=training_task_config.min_required_examples,
        gradient_clip_norm=training_task_config.gradient_clip_norm,
        deadline_at=training_task_config.deadline_at,
        notes=training_task_config.notes,
    )
