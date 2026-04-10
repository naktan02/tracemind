"""Training input backend package."""

from .base import (
    ANY_ADAPTER_KIND,
    PROTOTYPE_RESCORE_BACKEND_NAME,
    WEAK_STRONG_PAIR_BACKEND_NAME,
    TrainingExampleBackend,
    TrainingExampleBackendFactory,
)
from .models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
    TrainingExampleSource,
)
from .prototype_rescore import PrototypeRescoringTrainingExampleBackend
from .registry import (
    build_training_example_backend,
    register_training_example_backend,
    resolve_training_example_backend,
)
from .weak_strong_pair import WeakStrongPairTrainingExampleBackend

__all__ = [
    "ANY_ADAPTER_KIND",
    "PROTOTYPE_RESCORE_BACKEND_NAME",
    "PrototypeRescoringTrainingExampleBackend",
    "StoredEventTrainingExampleBuildRequest",
    "TrainingExampleBackend",
    "TrainingExampleBackendFactory",
    "TrainingExampleBuildRequest",
    "TrainingExampleSource",
    "WEAK_STRONG_PAIR_BACKEND_NAME",
    "WeakStrongPairTrainingExampleBackend",
    "build_training_example_backend",
    "register_training_example_backend",
    "resolve_training_example_backend",
]
