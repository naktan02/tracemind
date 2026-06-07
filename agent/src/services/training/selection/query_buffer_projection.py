"""Query buffer snapshot을 pseudo-label evidence로 투영하는 helper."""

from __future__ import annotations

import math
from collections.abc import Iterable

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
)
from methods.prototype.evidence.helpers import (
    build_ranked_evidence,
    rank_category_scores,
)
from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

QUERY_BUFFER_PROJECTION_BACKEND_NAME = "query_buffer_projection"
_FLOAT_TOLERANCE = 1e-9


def build_query_buffer_evidence(
    *,
    record: QueryBufferRecord,
    analysis_event: AnalysisEvent,
) -> PseudoLabelEvidence:
    """QueryBufferRecord와 AnalysisEvent를 공통 evidence로 정규화한다."""

    _validate_query_buffer_alignment(record=record, analysis_event=analysis_event)
    ranked_scores = rank_category_scores(analysis_event.category_scores)
    _validate_query_buffer_snapshot(record=record, ranked_scores=ranked_scores)
    return build_ranked_evidence(
        analysis_event=analysis_event,
        ranked_scores=ranked_scores,
        confidence_kind=record.confidence_kind,
        view_kind="single_view",
        backend_name=QUERY_BUFFER_PROJECTION_BACKEND_NAME,
        metadata=_build_projection_metadata(
            record=record,
            analysis_event=analysis_event,
        ),
    )


def build_query_buffer_evidences(
    *,
    records: Iterable[QueryBufferRecord],
    analysis_events: Iterable[AnalysisEvent],
) -> tuple[PseudoLabelEvidence, ...]:
    """QueryBufferRecord 순서를 유지하며 evidence 묶음을 만든다."""

    analysis_event_by_query_id: dict[str, AnalysisEvent] = {}
    for analysis_event in analysis_events:
        if analysis_event.query_id in analysis_event_by_query_id:
            raise ValueError(
                f"Duplicate analysis_event query_id: {analysis_event.query_id}."
            )
        analysis_event_by_query_id[analysis_event.query_id] = analysis_event

    evidences: list[PseudoLabelEvidence] = []
    for record in records:
        analysis_event = analysis_event_by_query_id.get(record.query_id)
        if analysis_event is None:
            raise ValueError(
                f"Missing AnalysisEvent for query buffer record: {record.query_id}."
            )
        evidences.append(
            build_query_buffer_evidence(
                record=record,
                analysis_event=analysis_event,
            )
        )
    return tuple(evidences)


def _validate_query_buffer_alignment(
    *,
    record: QueryBufferRecord,
    analysis_event: AnalysisEvent,
) -> None:
    if record.query_id != analysis_event.query_id:
        raise ValueError(
            "QueryBufferRecord query_id must match AnalysisEvent query_id."
        )
    if record.occurred_at != analysis_event.occurred_at:
        raise ValueError(
            "QueryBufferRecord occurred_at must match AnalysisEvent occurred_at."
        )


def _validate_query_buffer_snapshot(
    *,
    record: QueryBufferRecord,
    ranked_scores: list[tuple[str, float]],
) -> None:
    top1_label, top1_score = ranked_scores[0]
    if len(ranked_scores) > 1:
        top2_label, top2_score = ranked_scores[1]
    else:
        top2_label, top2_score = None, 0.0
    margin = top1_score - top2_score

    if record.predicted_label is not None and record.predicted_label != top1_label:
        raise ValueError("QueryBufferRecord predicted_label does not match scores.")
    if record.confidence is not None and not math.isclose(
        record.confidence,
        top1_score,
        rel_tol=_FLOAT_TOLERANCE,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError("QueryBufferRecord confidence does not match scores.")
    if record.runner_up_label is not None and record.runner_up_label != top2_label:
        raise ValueError("QueryBufferRecord runner_up_label does not match scores.")
    if record.runner_up_score is not None and not math.isclose(
        record.runner_up_score,
        top2_score,
        rel_tol=_FLOAT_TOLERANCE,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError("QueryBufferRecord runner_up_score does not match scores.")
    if record.margin is not None and not math.isclose(
        record.margin,
        margin,
        rel_tol=_FLOAT_TOLERANCE,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError("QueryBufferRecord margin does not match scores.")


def _build_projection_metadata(
    *,
    record: QueryBufferRecord,
    analysis_event: AnalysisEvent,
) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "query_buffer_schema_version": record.schema_version,
        "query_buffer_model_revision": record.model_revision,
        "query_buffer_locale": record.locale,
        "query_buffer_source_type": record.source_type,
        "translated_text_present": analysis_event.translated_text is not None,
    }
    for key, value in record.metadata.items():
        metadata[f"query_buffer.{key}"] = _coerce_metadata_scalar(value)
    return metadata


def _coerce_metadata_scalar(value: object) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    return str(value)
