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
    from methods.adaptation.local_update_registry import (
        build_shared_adapter_training_backend,
    )
    from methods.adaptation.text_classifier.peft_encoder.config import (
        LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
        LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
        PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    )
    from methods.adaptation.text_classifier.peft_encoder.training_backend import (
        PeftEncoderTrainingBackend,
    )
    from methods.adaptation.text_classifier.peft_encoder.update import (
        simulation_inline_delta,
    )

    backend = None
    objective_config = None if training_task is None else training_task.objective_config
    if objective_config is not None and objective_config.training_backend_name in {
        LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
        PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    }:
        candidate_backend = build_shared_adapter_training_backend(
            objective_config.training_backend_name,
            objective_config=objective_config,
        )
        if (
            isinstance(candidate_backend, PeftEncoderTrainingBackend)
            and candidate_backend.config.delta_format
            == LORA_CLASSIFIER_DELTA_FORMAT_INLINE
        ):
            backend = PeftEncoderTrainingBackend(
                backend_name=candidate_backend.backend_name,
                payload_format=candidate_backend.payload_format,
                adapter_kind=candidate_backend.adapter_kind,
                config=candidate_backend.config,
                train_executor=(
                    simulation_inline_delta.SimulationInlinePeftEncoderTrainExecutor()
                ),
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
