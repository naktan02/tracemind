"""Round request mapping helpers for federated simulation."""

from __future__ import annotations

from typing import Any

from main_server.src.services.federation.rounds.boundary.models import (
    RoundOpenDraftRequest,
    RoundTaskConfig,
)
from shared.src.contracts.model_contracts import ModelManifest


def build_federated_training_task_config(
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    min_required_examples: int,
    gradient_clip_norm: float | None,
    objective_config: Any,
    selection_policy: Any,
) -> Any:
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
    active_manifest: ModelManifest,
    round_id: str,
    training_task_config: Any,
) -> Any:
    """simulation task template을 canonical round open request로 변환한다."""

    del active_manifest
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
