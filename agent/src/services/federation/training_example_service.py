"""로컬 입력을 EmbeddedTrainingExample으로 변환하는 facade."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.services.training.input_backends.base import (
    TrainingExampleBackend,
)
from agent.src.services.training.input_backends.models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
    TrainingExampleSource,
)
from agent.src.services.training.input_backends.prototype_rescore import (
    PrototypeRescoringTrainingExampleBackend,
)
from agent.src.services.training.input_backends.registry import (
    build_training_example_backend,
    register_training_example_backend,
    resolve_training_example_backend,
)
from agent.src.services.training.input_backends.weak_strong_pair import (
    WeakStrongPairTrainingExampleBackend,
)
from agent.src.services.training.training_example_models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


@dataclass(slots=True)
class TrainingExampleService:
    """로컬 source row를 EmbeddedTrainingExample으로 변환한다."""

    backend: TrainingExampleBackend = field(
        default_factory=PrototypeRescoringTrainingExampleBackend
    )

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig,
    ) -> "TrainingExampleService":
        return cls(
            backend=resolve_training_example_backend(objective_config=objective_config)
        )

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        return self.backend.build_examples(request)

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        return self.backend.build_examples_from_stored_events(request)


__all__ = [
    "PrototypeRescoringTrainingExampleBackend",
    "StoredEventTrainingExampleBuildRequest",
    "TrainingExampleBackend",
    "TrainingExampleBuildRequest",
    "TrainingExampleService",
    "TrainingExampleSource",
    "WeakStrongPairTrainingExampleBackend",
    "build_training_example_backend",
    "register_training_example_backend",
    "resolve_training_example_backend",
]
