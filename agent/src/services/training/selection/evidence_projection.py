"""Analysis score를 pseudo-label evidence로 투영하는 공용 helper."""

from __future__ import annotations

from collections.abc import Mapping

from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PSEUDO_LABEL_EVIDENCE_V1,
    PseudoLabelEvidence,
)


def rank_category_scores(
    category_scores: Mapping[str, float],
) -> list[tuple[str, float]]:
    """category score를 confidence 내림차순, label 오름차순으로 정렬한다."""

    ranked_scores = sorted(
        ((str(label), float(score)) for label, score in category_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked_scores:
        raise ValueError("AnalysisEvent must contain at least one category score.")
    return ranked_scores


def build_ranked_evidence(
    *,
    analysis_event: AnalysisEvent,
    ranked_scores: list[tuple[str, float]],
    confidence_kind: str,
    view_kind: str,
    backend_name: str,
    sample_weight: float | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> PseudoLabelEvidence:
    """정렬된 category score를 canonical `PseudoLabelEvidence`로 변환한다."""

    top1_label, top1_score = ranked_scores[0]
    if len(ranked_scores) > 1:
        top2_label, top2_score = ranked_scores[1]
    else:
        top2_label, top2_score = None, 0.0
    return PseudoLabelEvidence(
        schema_version=PSEUDO_LABEL_EVIDENCE_V1,
        evidence_id=f"evidence:{analysis_event.query_id}",
        source_event_ref=analysis_event.query_id,
        occurred_at=analysis_event.occurred_at,
        label=top1_label,
        confidence=top1_score,
        confidence_kind=confidence_kind,
        margin=top1_score - top2_score,
        top1_label=top1_label,
        top1_score=top1_score,
        top2_label=top2_label,
        top2_score=top2_score,
        sample_weight=sample_weight if sample_weight is not None else top1_score,
        view_kind=view_kind,
        raw_scores=dict(analysis_event.category_scores),
        metadata={
            "evidence_backend_name": backend_name,
            "embedding_model_id": analysis_event.embedding_model_id,
            "translation_used": analysis_event.translation_model_id is not None,
            **({} if metadata is None else metadata),
        },
    )
