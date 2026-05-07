"""Evidence backend registry and resolver."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry_imports import (
    import_runtime_module_for_name,
    import_runtime_package_modules,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import (
    PseudoLabelEvidenceBackend,
    PseudoLabelEvidenceBackendFactory,
)

_PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY: dict[
    str,
    tuple[PseudoLabelEvidenceBackendFactory, RegistryCatalogEntry],
] = {}


def register_pseudo_label_evidence_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: PseudoLabelEvidenceBackendFactory | None = None,
) -> (
    Callable[[PseudoLabelEvidenceBackendFactory], PseudoLabelEvidenceBackendFactory]
    | PseudoLabelEvidenceBackendFactory
):
    """evidence backend factory 옆에서 runtime wiring을 등록한다."""

    def _decorator(
        factory: PseudoLabelEvidenceBackendFactory,
    ) -> PseudoLabelEvidenceBackendFactory:
        registered_backend = (factory, catalog_entry)
        for backend_name in backend_names:
            _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY[
                backend_name.strip().lower()
            ] = registered_backend
        return factory

    if factory is not None:
        return _decorator(factory)
    return _decorator


def build_pseudo_label_evidence_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
) -> PseudoLabelEvidenceBackend:
    """backend 이름과 objective config로 evidence backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    import_runtime_module_for_name(
        package_name="agent.src.services.training.backends.evidence",
        registered_name=normalized_name,
    )
    registered_backend = _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(objective_config)
    raise ValueError(f"Unsupported pseudo-label evidence backend: {backend_name}.")


def list_registered_pseudo_label_evidence_backend_names() -> tuple[str, ...]:
    """등록된 evidence backend 이름을 정렬된 tuple로 반환한다."""

    import_runtime_package_modules(
        package_name="agent.src.services.training.backends.evidence"
    )
    return tuple(sorted(_PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY))


def list_pseudo_label_evidence_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 evidence backend catalog entry를 canonical item 기준으로 반환한다."""

    import_runtime_package_modules(
        package_name="agent.src.services.training.backends.evidence"
    )
    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.values()
    )
