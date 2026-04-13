"""Pseudo-label evidence backend package."""

from .base import (
    ANY_ADAPTER_KIND,
    PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    PseudoLabelEvidenceBackend,
    PseudoLabelEvidenceBackendFactory,
)
from .prototype_similarity import PrototypeSimilarityEvidenceBackend
from .registry import (
    build_pseudo_label_evidence_backend,
    register_pseudo_label_evidence_backend,
    resolve_pseudo_label_evidence_backend,
)

__all__ = [
    "ANY_ADAPTER_KIND",
    "PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME",
    "PrototypeSimilarityEvidenceBackend",
    "PseudoLabelEvidenceBackend",
    "PseudoLabelEvidenceBackendFactory",
    "build_pseudo_label_evidence_backend",
    "register_pseudo_label_evidence_backend",
    "resolve_pseudo_label_evidence_backend",
]
