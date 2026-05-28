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

    backend = None
    objective_config = None if training_task is None else training_task.objective_config
    if objective_config is not None:
        backend = build_shared_adapter_training_backend(
            objective_config.training_backend_name,
            objective_config=objective_config,
        )
        inline_executor_builder = getattr(
            backend,
            "with_simulation_inline_train_executor",
            None,
        )
        if callable(inline_executor_builder):
            backend = inline_executor_builder()

    return LocalTrainingService(
        repository=TrainingArtifactRepository(state_root=client_state_root),
        backend=backend,
    )


def build_query_ssl_local_training_service(
    *,
    client_state_root: Path,
    backend: Any | None = None,
) -> Any:
    """simulation client state root 기준 Query SSL local training service를 만든다."""

    from agent.src.infrastructure.repositories.training_artifact_repository import (
        TrainingArtifactRepository,
    )
    from agent.src.services.training.execution.query_ssl_local_training_service import (
        QuerySslLocalTrainingService,
    )

    return QuerySslLocalTrainingService(
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
