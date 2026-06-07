"""로컬 입력을 EmbeddedTrainingExample으로 변환하는 facade."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.training.backends.inputs.base import (
    TrainingExampleBackend,
)
from agent.src.services.training.backends.inputs.models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
)
from agent.src.services.training.backends.inputs.resolver import (
    resolve_training_example_backend,
)
from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


@dataclass(slots=True)
class TrainingExampleService:
    """로컬 source row를 EmbeddedTrainingExample으로 변환한다."""

    backend: TrainingExampleBackend | None = None

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
        return self._require_backend().build_examples(request)

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        return self._require_backend().build_examples_from_stored_events(request)

    def _require_backend(self) -> TrainingExampleBackend:
        if self.backend is None:
            raise ValueError(
                "TrainingExampleService requires an explicit backend. "
                "Use from_objective_config() for runtime config selection."
            )
        return self.backend
