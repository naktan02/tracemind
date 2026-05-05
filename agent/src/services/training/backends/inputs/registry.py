"""Training input backend registry and resolver."""

from __future__ import annotations

from agent.src.services.training.backends.training.base import (
    SharedAdapterTrainingBackend,
)
from agent.src.services.training.backends.training.registry import (
    build_shared_adapter_training_backend,
)
from shared.src.config.local_training_registry_catalog import (
    PROTOTYPE_RESCORE_EXAMPLE_BACKEND_CATALOG_ENTRY,
    WEAK_STRONG_PAIR_EXAMPLE_BACKEND_CATALOG_ENTRY,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import (
    ANY_ADAPTER_KIND,
    PROTOTYPE_RESCORE_BACKEND_NAME,
    WEAK_STRONG_PAIR_BACKEND_NAME,
    TrainingExampleBackend,
    TrainingExampleBackendFactory,
)
from .prototype_rescore import PrototypeRescoringTrainingExampleBackend
from .weak_strong_pair import WeakStrongPairTrainingExampleBackend

_TRAINING_EXAMPLE_BACKEND_REGISTRY: dict[
    str,
    tuple[TrainingExampleBackendFactory, RegistryCatalogEntry],
] = {}


def register_training_example_backend(
    *backend_names: str,
    factory: TrainingExampleBackendFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """얇은 wiring registry에 training example backend를 등록한다."""

    registered_backend = (factory, catalog_entry)
    for backend_name in backend_names:
        _TRAINING_EXAMPLE_BACKEND_REGISTRY[backend_name.strip().lower()] = (
            registered_backend
        )


def build_training_example_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
) -> TrainingExampleBackend:
    """backend 이름과 objective config로 training example backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    registered_backend = _TRAINING_EXAMPLE_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(objective_config)
    raise ValueError(f"Unsupported training example backend: {backend_name}.")


def list_registered_training_example_backend_names() -> tuple[str, ...]:
    """등록된 training example backend 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_TRAINING_EXAMPLE_BACKEND_REGISTRY))


def list_training_example_backend_catalog_entries() -> tuple[RegistryCatalogEntry, ...]:
    """등록된 example backend catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _TRAINING_EXAMPLE_BACKEND_REGISTRY.values()
    )


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
    _require_adapter_kind_support(
        component_type="training example backend",
        component_name=backend.backend_name,
        supported_adapter_kinds=backend.supported_adapter_kinds,
        adapter_kind=resolved_training_backend.adapter_kind,
    )
    return backend


def _require_adapter_kind_support(
    *,
    component_type: str,
    component_name: str,
    supported_adapter_kinds: tuple[str, ...],
    adapter_kind: str,
) -> None:
    normalized_supported = tuple(
        value.strip().lower() for value in supported_adapter_kinds
    )
    normalized_adapter_kind = adapter_kind.strip().lower()
    if (
        ANY_ADAPTER_KIND in normalized_supported
        or normalized_adapter_kind in normalized_supported
    ):
        return
    raise ValueError(
        f"Incompatible {component_type}: {component_name} does not support "
        f"adapter_kind={adapter_kind}."
    )


register_training_example_backend(
    PROTOTYPE_RESCORE_BACKEND_NAME,
    factory=lambda _objective_config: PrototypeRescoringTrainingExampleBackend(),
    catalog_entry=PROTOTYPE_RESCORE_EXAMPLE_BACKEND_CATALOG_ENTRY,
)
register_training_example_backend(
    WEAK_STRONG_PAIR_BACKEND_NAME,
    factory=lambda _objective_config: WeakStrongPairTrainingExampleBackend(),
    catalog_entry=WEAK_STRONG_PAIR_EXAMPLE_BACKEND_CATALOG_ENTRY,
)
