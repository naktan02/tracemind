"""Evidence backend registry and resolver."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.services.runtime_registry import RuntimeRegistry
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import (
    PseudoLabelEvidenceBackend,
    PseudoLabelEvidenceBackendFactory,
)

_PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY = RuntimeRegistry[
    PseudoLabelEvidenceBackendFactory
](
    package_name="agent.src.services.training.backends.evidence",
    item_kind="pseudo-label evidence backend",
)


def register_pseudo_label_evidence_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: PseudoLabelEvidenceBackendFactory | None = None,
) -> (
    Callable[[PseudoLabelEvidenceBackendFactory], PseudoLabelEvidenceBackendFactory]
    | PseudoLabelEvidenceBackendFactory
):
    """evidence backend factory 옆에서 runtime wiring을 등록한다."""

    return _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.register(
        *backend_names,
        catalog_entry=catalog_entry,
        factory=factory,
    )


def build_pseudo_label_evidence_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
) -> PseudoLabelEvidenceBackend:
    """backend 이름과 objective config로 evidence backend를 조립한다."""

    factory, _catalog_entry = _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.get(backend_name)
    return factory(objective_config)


def list_registered_pseudo_label_evidence_backend_names() -> tuple[str, ...]:
    """등록된 evidence backend 이름을 정렬된 tuple로 반환한다."""

    return _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.list_names()


def list_pseudo_label_evidence_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 evidence backend catalog entry를 canonical item 기준으로 반환한다."""

    return _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.list_catalog_entries()
