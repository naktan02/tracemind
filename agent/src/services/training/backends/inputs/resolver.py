"""Training example backend resolver."""

from __future__ import annotations

from methods.adaptation.local_update_backend import (
    SharedAdapterTrainingBackend,
)
from methods.adaptation.local_update_registry import (
    build_shared_adapter_training_backend,
)
from methods.federated_ssl.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import TrainingExampleBackend
from .compatibility import require_training_example_backend_adapter_kind_support
from .registry import build_training_example_backend


def resolve_training_example_backend(
    *,
    objective_config: TrainingObjectiveConfig,
    training_backend: SharedAdapterTrainingBackend | None = None,
) -> TrainingExampleBackend:
    """objective config 기준으로 example backend를 검증해 조립한다."""

    backend_name = (
        objective_config.example_generation_backend_name
        or DEFAULT_TRAINING_PROFILE.example_generation_backend_name
    )
    backend = build_training_example_backend(
        backend_name,
        objective_config=objective_config,
    )
    resolved_training_backend = (
        training_backend
        or build_shared_adapter_training_backend(
            objective_config.training_backend_name,
            objective_config=objective_config,
        )
    )
    require_training_example_backend_adapter_kind_support(
        backend=backend,
        adapter_kind=resolved_training_backend.adapter_kind,
    )
    return backend
