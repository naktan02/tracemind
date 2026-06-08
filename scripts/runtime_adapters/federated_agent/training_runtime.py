"""FL simulation용 agent local training runtime bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_query_ssl_local_training_service(
    *,
    client_state_root: Path,
    backend: Any | None = None,
) -> Any:
    """simulation client state root 기준 Query SSL local training service를 만든다."""

    from agent.src.infrastructure.repositories.training_artifact_repository import (
        TrainingArtifactRepository,
    )
    from agent.src.services.training_runtime.query_ssl_peft.local_training_service import (  # noqa: E501
        QuerySslLocalTrainingService,
    )

    return QuerySslLocalTrainingService(
        repository=TrainingArtifactRepository(state_root=client_state_root),
        backend=backend,
    )
