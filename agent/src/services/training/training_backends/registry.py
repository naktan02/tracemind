"""Training backend registry and resolver."""

from __future__ import annotations

from .base import SharedAdapterTrainingBackend, TrainingBackendFactory
from .diagonal_scale_heuristic import DiagonalScaleHeuristicTrainingBackend

_TRAINING_BACKEND_REGISTRY: dict[str, TrainingBackendFactory] = {}


def register_shared_adapter_training_backend(
    *backend_names: str,
    factory: TrainingBackendFactory,
) -> None:
    """얇은 wiring registry에 backend factory를 등록한다."""

    for backend_name in backend_names:
        _TRAINING_BACKEND_REGISTRY[backend_name.strip().lower()] = factory


def build_shared_adapter_training_backend(
    backend_name: str,
    *,
    objective_config=None,
) -> SharedAdapterTrainingBackend:
    """backend 이름으로 로컬 학습 backend를 생성한다."""

    normalized_name = backend_name.strip().lower()
    factory = _TRAINING_BACKEND_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory(objective_config)
    raise ValueError(f"Unsupported local training backend: {backend_name}.")


def list_registered_shared_adapter_training_backend_names() -> tuple[str, ...]:
    """등록된 로컬 training backend 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_TRAINING_BACKEND_REGISTRY))


register_shared_adapter_training_backend(
    "diagonal_scale_heuristic",
    "synthetic_vector_adapter",
    factory=DiagonalScaleHeuristicTrainingBackend.from_objective_config,
)


__all__ = [
    "build_shared_adapter_training_backend",
    "list_registered_shared_adapter_training_backend_names",
    "register_shared_adapter_training_backend",
]
