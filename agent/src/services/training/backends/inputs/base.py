"""Training input backend protocol and constants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.contracts.training_example_backends import (
    WEAK_STRONG_PAIR_EXAMPLE_BACKEND,
)

from .models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
)

ANY_ADAPTER_KIND = "*"
WEAK_STRONG_PAIR_BACKEND_NAME = WEAK_STRONG_PAIR_EXAMPLE_BACKEND


class TrainingExampleBackend(Protocol):
    """학습 예시 재구성 backend 인터페이스."""

    backend_name: str
    supported_adapter_kinds: tuple[str, ...]
    supports_stored_event_rebuild: bool

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        """source row에서 학습 예시를 만든다."""

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]:
        """stored event에서 학습 예시를 재구성한다."""


TrainingExampleBackendFactory = Callable[
    [TrainingObjectiveConfig],
    TrainingExampleBackend,
]
