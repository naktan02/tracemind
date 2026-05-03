"""Evidence backend registry and resolver."""

from __future__ import annotations

from shared.src.config.local_training_registry_catalog import (
    PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_CATALOG_ENTRY,
)
from shared.src.config.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import (
    PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    PseudoLabelEvidenceBackend,
    PseudoLabelEvidenceBackendFactory,
)
from .prototype_similarity import PrototypeSimilarityEvidenceBackend

_PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY: dict[
    str,
    tuple[PseudoLabelEvidenceBackendFactory, RegistryCatalogEntry],
] = {}


def register_pseudo_label_evidence_backend(
    *backend_names: str,
    factory: PseudoLabelEvidenceBackendFactory,
    catalog_entry: RegistryCatalogEntry,
) -> None:
    """얇은 wiring registry에 evidence backend를 등록한다."""

    registered_backend = (factory, catalog_entry)
    for backend_name in backend_names:
        _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY[backend_name.strip().lower()] = (
            registered_backend
        )


def build_pseudo_label_evidence_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
) -> PseudoLabelEvidenceBackend:
    """backend 이름과 objective config로 evidence backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    registered_backend = _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.get(normalized_name)
    if registered_backend is not None:
        factory, _catalog_entry = registered_backend
        return factory(objective_config)
    raise ValueError(f"Unsupported pseudo-label evidence backend: {backend_name}.")


def list_registered_pseudo_label_evidence_backend_names() -> tuple[str, ...]:
    """등록된 evidence backend 이름을 정렬된 tuple로 반환한다."""

    return tuple(sorted(_PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY))


def list_pseudo_label_evidence_backend_catalog_entries(
) -> tuple[RegistryCatalogEntry, ...]:
    """등록된 evidence backend catalog entry를 canonical item 기준으로 반환한다."""

    return dedupe_registry_catalog_entries(
        catalog_entry
        for _factory, catalog_entry in _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.values()
    )


def resolve_pseudo_label_evidence_backend(
    *,
    objective_config: TrainingObjectiveConfig,
) -> PseudoLabelEvidenceBackend:
    """objective config 기준으로 evidence backend를 조립한다."""

    backend_name = (
        objective_config.evidence_backend_name
        or DEFAULT_TRAINING_PROFILE.evidence_backend_name
    )
    return build_pseudo_label_evidence_backend(
        backend_name,
        objective_config=objective_config,
    )


register_pseudo_label_evidence_backend(
    PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    factory=lambda _objective_config: PrototypeSimilarityEvidenceBackend(),
    catalog_entry=PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_CATALOG_ENTRY,
)
