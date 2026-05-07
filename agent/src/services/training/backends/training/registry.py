"""Training backend registry and resolver."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry import RuntimeRegistry
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry

from .base import SharedAdapterTrainingBackend, TrainingBackendFactory

_TRAINING_BACKEND_REGISTRY = RuntimeRegistry[TrainingBackendFactory](
    package_name="agent.src.services.training.backends.training",
    item_kind="local training backend",
)


def register_shared_adapter_training_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: TrainingBackendFactory | None = None,
) -> (
    Callable[[TrainingBackendFactory], TrainingBackendFactory]
    | TrainingBackendFactory
):
    """training backend factory 옆에서 runtime wiring을 등록한다."""

    return _TRAINING_BACKEND_REGISTRY.register(
        *backend_names,
        catalog_entry=catalog_entry,
        factory=factory,
    )


def build_shared_adapter_training_backend(
    backend_name: str,
    *,
    objective_config=None,
) -> SharedAdapterTrainingBackend:
    """backend 이름으로 로컬 학습 backend를 생성한다."""

    factory, _catalog_entry = _TRAINING_BACKEND_REGISTRY.get(backend_name)
    return factory(objective_config)


def list_registered_shared_adapter_training_backend_names() -> tuple[str, ...]:
    """등록된 로컬 training backend 이름을 정렬된 tuple로 반환한다."""

    return _TRAINING_BACKEND_REGISTRY.list_names()


def list_shared_adapter_training_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 training backend catalog entry를 canonical item 기준으로 반환한다."""

    return _TRAINING_BACKEND_REGISTRY.list_catalog_entries()
