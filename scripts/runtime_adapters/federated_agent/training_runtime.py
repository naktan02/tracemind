"""FL simulation용 agent local training runtime bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_federated_local_training_service(
    *,
    client_state_root: Path,
) -> Any:
    """simulation client state root 기준 local training service를 만든다."""

    from agent.src.infrastructure.repositories.training_artifact_repository import (
        TrainingArtifactRepository,
    )
    from agent.src.services.training.execution.local_training_service import (
        LocalTrainingService,
    )

    return LocalTrainingService(
        repository=TrainingArtifactRepository(state_root=client_state_root)
    )


def run_federated_local_training(
    *,
    local_training_service: Any,
    training_examples: tuple[Any, ...],
    training_task: Any,
    model_manifest: Any,
) -> Any:
    """LocalTrainingRequest 생성 세부사항을 scripts runtime bridge에 숨긴다."""

    from agent.src.services.training.execution.local_training_service import (
        LocalTrainingRequest,
    )

    return local_training_service.run(
        LocalTrainingRequest(
            training_examples=training_examples,
            training_task=training_task,
            model_manifest=model_manifest,
        )
    )
