"""Pseudo-label evidence backend resolver."""

from __future__ import annotations

from agent.src.services.training.selection.stored_event_defaults import (
    STORED_EVENT_EVIDENCE_BACKEND_NAME,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

from .base import PseudoLabelEvidenceBackend
from .registry import build_pseudo_label_evidence_backend


def resolve_pseudo_label_evidence_backend(
    *,
    objective_config: TrainingObjectiveConfig,
) -> PseudoLabelEvidenceBackend:
    """objective config 기준으로 evidence backend를 조립한다."""

    backend_name = (
        objective_config.evidence_backend_name or STORED_EVENT_EVIDENCE_BACKEND_NAME
    )
    return build_pseudo_label_evidence_backend(
        backend_name,
        objective_config=objective_config,
    )
