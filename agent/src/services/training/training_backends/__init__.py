"""Shared adapter training backend package."""

from .base import (
    AcceptedTrainingExample,
    SharedAdapterTrainingBackend,
    TrainingBackend,
    TrainingBackendFactory,
)
from .diagonal_scale_heuristic import (
    DiagonalScaleHeuristicTrainingBackend,
    SyntheticVectorAdapterTrainingBackend,
    build_diagonal_scale_heuristic_training_backend_config,
)
from .registry import (
    build_shared_adapter_training_backend,
    register_shared_adapter_training_backend,
)

__all__ = [
    "AcceptedTrainingExample",
    "build_diagonal_scale_heuristic_training_backend_config",
    "build_shared_adapter_training_backend",
    "DiagonalScaleHeuristicTrainingBackend",
    "register_shared_adapter_training_backend",
    "SharedAdapterTrainingBackend",
    "SyntheticVectorAdapterTrainingBackend",
    "TrainingBackend",
    "TrainingBackendFactory",
]
