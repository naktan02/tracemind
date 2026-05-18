"""FL simulation용 agent local training runtime bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_federated_local_training_service(
    *,
    client_state_root: Path,
    training_task: Any | None = None,
) -> Any:
    """simulation client state root 기준 local training service를 만든다."""

    from agent.src.infrastructure.repositories.training_artifact_repository import (
        TrainingArtifactRepository,
    )
    from agent.src.services.training.execution.local_training_service import (
        LocalTrainingService,
    )
    from methods.adaptation.lora_classifier.config import (
        LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
        LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
        build_lora_classifier_training_backend_config,
    )
    from methods.adaptation.lora_classifier.training_backend import (
        LoraClassifierTrainingBackend,
    )
    from methods.adaptation.lora_classifier.update.simulation_inline_delta import (
        SimulationInlineLoraClassifierTrainExecutor,
    )

    backend = None
    objective_config = None if training_task is None else training_task.objective_config
    if (
        objective_config is not None
        and objective_config.training_backend_name
        == LORA_CLASSIFIER_TRAINING_BACKEND_NAME
    ):
        lora_config = build_lora_classifier_training_backend_config(objective_config)
        if lora_config.delta_format == LORA_CLASSIFIER_DELTA_FORMAT_INLINE:
            backend = LoraClassifierTrainingBackend(
                config=lora_config,
                train_executor=SimulationInlineLoraClassifierTrainExecutor(),
            )

    return LocalTrainingService(
        repository=TrainingArtifactRepository(state_root=client_state_root),
        backend=backend,
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
