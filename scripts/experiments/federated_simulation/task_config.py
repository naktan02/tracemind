"""Federated simulation용 task/config 변환 helper."""

from __future__ import annotations

from main_server.src.services.rounds.models import RoundOpenRequest
from shared.src.contracts.model_contracts import ModelManifest

from .models import FederatedTrainingTaskConfig


def build_round_open_request(
    *,
    active_manifest: ModelManifest,
    round_id: str,
    training_task_config: FederatedTrainingTaskConfig,
) -> RoundOpenRequest:
    """simulation training task 설정을 round open request로 변환한다."""
    return RoundOpenRequest(
        active_manifest=active_manifest,
        round_id=round_id,
        local_epochs=int(training_task_config.local_epochs),
        batch_size=int(training_task_config.batch_size),
        learning_rate=float(training_task_config.learning_rate),
        max_steps=int(training_task_config.max_steps),
        objective_config=training_task_config.objective_config.to_mapping(),
        selection_policy=training_task_config.selection_policy.to_mapping(),
        min_required_examples=int(training_task_config.min_required_examples),
        gradient_clip_norm=(
            None
            if training_task_config.gradient_clip_norm is None
            else float(training_task_config.gradient_clip_norm)
        ),
    )
