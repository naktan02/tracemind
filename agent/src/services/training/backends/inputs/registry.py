"""Training input backend registry and resolver."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry import RuntimeRegistry
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import (
    TrainingExampleBackend,
    TrainingExampleBackendFactory,
)

_TRAINING_EXAMPLE_BACKEND_REGISTRY = RuntimeRegistry[TrainingExampleBackendFactory](
    package_name="agent.src.services.training.backends.inputs",
    item_kind="training example backend",
)


def register_training_example_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: TrainingExampleBackendFactory | None = None,
) -> (
    Callable[[TrainingExampleBackendFactory], TrainingExampleBackendFactory]
    | TrainingExampleBackendFactory
):
    """example backend factory 옆에서 runtime wiring을 등록한다."""

    return _TRAINING_EXAMPLE_BACKEND_REGISTRY.register(
        *backend_names,
        catalog_entry=catalog_entry,
        factory=factory,
    )


def build_training_example_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
) -> TrainingExampleBackend:
    """backend 이름과 objective config로 training example backend를 조립한다."""

    factory, _catalog_entry = _TRAINING_EXAMPLE_BACKEND_REGISTRY.get(backend_name)
    return factory(objective_config)


def list_registered_training_example_backend_names() -> tuple[str, ...]:
    """등록된 training example backend 이름을 정렬된 tuple로 반환한다."""

    return _TRAINING_EXAMPLE_BACKEND_REGISTRY.list_names()


def list_training_example_backend_catalog_entries() -> tuple[RegistryCatalogEntry, ...]:
    """등록된 example backend catalog entry를 canonical item 기준으로 반환한다."""

    return _TRAINING_EXAMPLE_BACKEND_REGISTRY.list_catalog_entries()
