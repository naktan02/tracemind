"""Training input backend protocol and constants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from agent.src.services.training.training_example_models import (
    EmbeddedTrainingExample,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .models import (
    StoredEventTrainingExampleBuildRequest,
    TrainingExampleBuildRequest,
)

ANY_ADAPTER_KIND = "*"
PROTOTYPE_RESCORE_BACKEND_NAME = "prototype_rescore"
WEAK_STRONG_PAIR_BACKEND_NAME = "weak_strong_pair"


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


__all__ = [
    "ANY_ADAPTER_KIND",
    "PROTOTYPE_RESCORE_BACKEND_NAME",
    "TrainingExampleBackend",
    "TrainingExampleBackendFactory",
    "WEAK_STRONG_PAIR_BACKEND_NAME",
]
