"""Evidence backend helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PSEUDO_LABEL_EVIDENCE_V1,
    PseudoLabelEvidence,
)


def rank_category_scores(
    category_scores: Mapping[str, float],
) -> list[tuple[str, float]]:
    ranked_scores = sorted(
        ((str(label), float(score)) for label, score in category_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked_scores:
        raise ValueError("ScoredEvent must contain at least one category score.")
    return ranked_scores


def build_ranked_evidence(
    *,
    scored_event: ScoredEvent,
    ranked_scores: list[tuple[str, float]],
    confidence_kind: str,
    view_kind: str,
    backend_name: str,
    label_distribution: dict[str, float] | None = None,
    sample_weight: float | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> PseudoLabelEvidence:
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
        confidence_kind=confidence_kind,
        margin=margin,
        top1_label=top1_label,
        top1_score=top1_score,
        top2_label=top2_label,
        top2_score=top2_score,
        sample_weight=sample_weight if sample_weight is not None else top1_score,
        view_kind=view_kind,
        raw_scores=dict(scored_event.category_scores),
        label_distribution=label_distribution,
        metadata={
            "evidence_backend_name": backend_name,
            "embedding_model_id": scored_event.embedding_model_id,
            "translation_used": scored_event.translation_model_id is not None,
            **({} if metadata is None else metadata),
        },
    )


def softmax_distribution(
    category_scores: Mapping[str, float],
    *,
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    scaled_pairs = [(label, float(score) / temperature) for label, score in category_scores.items()]
    if not scaled_pairs:
        raise ValueError("ScoredEvent must contain at least one category score.")
    max_score = max(score for _, score in scaled_pairs)
    exp_pairs = [
        (label, math.exp(score - max_score))
        for label, score in scaled_pairs
    ]
    normalizer = math.fsum(value for _, value in exp_pairs)
    return {label: value / normalizer for label, value in exp_pairs}


__all__ = [
    "build_ranked_evidence",
    "rank_category_scores",
    "softmax_distribution",
]
