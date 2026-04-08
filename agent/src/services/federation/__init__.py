"""연합학습 runtime 조합 서비스."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .round_client import RoundClient
from .training_example_service import (
    PrototypeRescoringTrainingExampleBackend,
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBackend,
    TrainingExampleBuildRequest,
    TrainingExampleService,
    TrainingExampleSource,
    build_training_example_backend,
    register_training_example_backend,
)

if TYPE_CHECKING:
    from .runtime_service import (
        FederationRunResult,
        FederationRunStatus,
        FederationRuntimeService,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "FederationRunResult",
        "FederationRunStatus",
        "FederationRuntimeService",
    }:
        from . import runtime_service

        return getattr(runtime_service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FederationRunResult",
    "FederationRunStatus",
    "PrototypeRescoringTrainingExampleBackend",
    "FederationRuntimeService",
    "RoundClient",
    "StoredEventTrainingExampleBuildRequest",
    "TrainingExampleBackend",
    "TrainingExampleBuildRequest",
    "TrainingExampleService",
    "TrainingExampleSource",
    "build_training_example_backend",
    "register_training_example_backend",
]
