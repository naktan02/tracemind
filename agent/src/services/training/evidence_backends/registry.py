"""Evidence backend registry and resolver."""

from __future__ import annotations

from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import (
    FIXMATCH_WEAK_VIEW_EVIDENCE_BACKEND_NAME,
    PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    PseudoLabelEvidenceBackend,
    PseudoLabelEvidenceBackendFactory,
)
from .fixmatch import FixMatchWeakViewEvidenceBackend
from .prototype_similarity import PrototypeSimilarityEvidenceBackend

_PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY: dict[
    str, PseudoLabelEvidenceBackendFactory
] = {}


def register_pseudo_label_evidence_backend(
    *backend_names: str,
    factory: PseudoLabelEvidenceBackendFactory,
) -> None:
    """얇은 wiring registry에 evidence backend를 등록한다."""

    for backend_name in backend_names:
        _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY[backend_name.strip().lower()] = (
            factory
        )


def build_pseudo_label_evidence_backend(
    backend_name: str,
    *,
    objective_config: TrainingObjectiveConfig,
) -> PseudoLabelEvidenceBackend:
    """backend 이름과 objective config로 evidence backend를 조립한다."""

    normalized_name = backend_name.strip().lower()
    factory = _PSEUDO_LABEL_EVIDENCE_BACKEND_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory(objective_config)
    raise ValueError(f"Unsupported pseudo-label evidence backend: {backend_name}.")


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
)
register_pseudo_label_evidence_backend(
    FIXMATCH_WEAK_VIEW_EVIDENCE_BACKEND_NAME,
    factory=FixMatchWeakViewEvidenceBackend.from_objective_config,
)


__all__ = [
    "build_pseudo_label_evidence_backend",
    "register_pseudo_label_evidence_backend",
    "resolve_pseudo_label_evidence_backend",
]
