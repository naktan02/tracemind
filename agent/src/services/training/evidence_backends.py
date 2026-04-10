"""Pseudo-label evidence backend 구현과 resolver."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PSEUDO_LABEL_EVIDENCE_V1,
    PseudoLabelEvidence,
)

ANY_ADAPTER_KIND = "*"
PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME = "prototype_similarity_evidence"


class PseudoLabelEvidenceBackend(Protocol):
    """ScoredEvent를 공통 pseudo-label evidence로 정규화한다."""

    backend_name: str
    supported_adapter_kinds: tuple[str, ...]

    def build_evidence(
        self,
        *,
        scored_event: ScoredEvent,
    ) -> PseudoLabelEvidence:
        """ScoredEvent 하나를 evidence 하나로 변환한다."""


PseudoLabelEvidenceBackendFactory = Callable[
    [TrainingObjectiveConfig],
    PseudoLabelEvidenceBackend,
]


@dataclass(slots=True)
class PrototypeSimilarityEvidenceBackend:
    """현재 prototype similarity score를 공통 evidence로 정규화한다."""

    backend_name: str = PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME
    supported_adapter_kinds: tuple[str, ...] = (ANY_ADAPTER_KIND,)
    confidence_kind: str = "prototype_similarity"
    view_kind: str = "single_view"

    def build_evidence(
        self,
        *,
        scored_event: ScoredEvent,
    ) -> PseudoLabelEvidence:
        ranked_scores = _rank_category_scores(scored_event.category_scores)
        top1_label, top1_score = ranked_scores[0]
        if len(ranked_scores) > 1:
            top2_label, top2_score = ranked_scores[1]
        else:
            top2_label, top2_score = None, 0.0
        margin = top1_score - top2_score
        return PseudoLabelEvidence(
            schema_version=PSEUDO_LABEL_EVIDENCE_V1,
            evidence_id=f"evidence:{scored_event.query_id}",
            source_event_ref=scored_event.query_id,
            occurred_at=scored_event.occurred_at,
            label=top1_label,
            confidence=top1_score,
            confidence_kind=self.confidence_kind,
            margin=margin,
            top1_label=top1_label,
            top1_score=top1_score,
            top2_label=top2_label,
            top2_score=top2_score,
            sample_weight=top1_score,
            view_kind=self.view_kind,
            raw_scores=dict(scored_event.category_scores),
            label_distribution=None,
            metadata={
                "evidence_backend_name": self.backend_name,
                "embedding_model_id": scored_event.embedding_model_id,
                "translation_used": scored_event.translation_model_id is not None,
            },
        )


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


def _rank_category_scores(
    category_scores: Mapping[str, float],
) -> list[tuple[str, float]]:
    ranked_scores = sorted(
        category_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked_scores:
        raise ValueError("ScoredEvent must contain at least one category score.")
    return ranked_scores


register_pseudo_label_evidence_backend(
    PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME,
    factory=lambda _objective_config: PrototypeSimilarityEvidenceBackend(),
)


__all__ = [
    "PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_NAME",
    "PrototypeSimilarityEvidenceBackend",
    "PseudoLabelEvidenceBackend",
    "build_pseudo_label_evidence_backend",
    "register_pseudo_label_evidence_backend",
    "resolve_pseudo_label_evidence_backend",
]
