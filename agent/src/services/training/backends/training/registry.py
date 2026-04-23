"""Training backend registry and resolver."""

from __future__ import annotations

from shared.src.config.adapter_family_metadata import DIAGONAL_SCALE_FAMILY_METADATA
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

from .base import SharedAdapterTrainingBackend, TrainingBackendFactory
from .diagonal_scale_heuristic import DiagonalScaleHeuristicTrainingBackend

_TRAINING_BACKEND_REGISTRY: dict[
    str,
    tuple[TrainingBackendFactory, RegistryCatalogEntry],
] = {}


def register_shared_adapter_training_backend(
    *backend_names: str,
    factory: TrainingBackendFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """얇은 wiring registry에 backend factory를 등록한다."""

    registered_backend = (factory, catalog_entry)
    for backend_name in backend_names:
        _TRAINING_BACKEND_REGISTRY[
            backend_name.strip().lower()
        ] = registered_backend


def build_shared_adapter_training_backend(
    backend_name: str,
    *,
    objective_config=None,
) -> SharedAdapterTrainingBackend:
    """backend 이름으로 로컬 학습 backend를 생성한다."""

    normalized_name = backend_name.strip().lower()
    registered_backend = _TRAINING_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(objective_config)
    raise ValueError(f"Unsupported local training backend: {backend_name}.")


def list_registered_shared_adapter_training_backend_names() -> tuple[str, ...]:
    """등록된 로컬 training backend 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_TRAINING_BACKEND_REGISTRY))


def list_shared_adapter_training_backend_catalog_entries(
) -> tuple[RegistryCatalogEntry, ...]:
    """등록된 training backend catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _TRAINING_BACKEND_REGISTRY.values()
    )


register_shared_adapter_training_backend(
    "diagonal_scale_heuristic",
    "synthetic_vector_adapter",
    factory=DiagonalScaleHeuristicTrainingBackend.from_objective_config,
    catalog_entry=RegistryCatalogEntry(
        item_name="diagonal_scale_heuristic",
        display_name="diagonal_scale_heuristic",
        implementation_module=DiagonalScaleHeuristicTrainingBackend.__module__,
        core_method_name="diagonal_scale_heuristic",
        family_name=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
        supported_adapter_kinds=(DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,),
        accepted_payload_formats=(
            DIAGONAL_SCALE_FAMILY_METADATA.canonical_update_payload_format,
        ),
        metadata={
            "payload_format": (
                DIAGONAL_SCALE_FAMILY_METADATA.canonical_update_payload_format
            )
        },
    ),
)


__all__ = [
    "build_shared_adapter_training_backend",
    "list_shared_adapter_training_backend_catalog_entries",
    "list_registered_shared_adapter_training_backend_names",
    "register_shared_adapter_training_backend",
]
