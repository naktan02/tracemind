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
from .fixmatch_consistency import (
    ClassifierHeadFixMatchConsistencyTrainingBackend,
    build_classifier_head_fixmatch_training_backend_config,
)
from .registry import (
    build_shared_adapter_training_backend,
    register_shared_adapter_training_backend,
)

__all__ = [
    "AcceptedTrainingExample",
    "build_classifier_head_fixmatch_training_backend_config",
    "build_diagonal_scale_heuristic_training_backend_config",
    "build_shared_adapter_training_backend",
    "ClassifierHeadFixMatchConsistencyTrainingBackend",
    "DiagonalScaleHeuristicTrainingBackend",
    "register_shared_adapter_training_backend",
    "SharedAdapterTrainingBackend",
    "SyntheticVectorAdapterTrainingBackend",
    "TrainingBackend",
    "TrainingBackendFactory",
]
