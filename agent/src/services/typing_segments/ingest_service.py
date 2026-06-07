"""Typing segment를 agent-local inference 입력으로 변환하는 service."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.inference.pipeline_service import (
    InferencePipelineResult,
    InferencePipelineService,
)
from shared.src.contracts.typing_segment_contracts import (
    TypingSegmentBatchIngestResponsePayload,
    TypingSegmentIngestResponsePayload,
    TypingSegmentPayload,
)
from shared.src.domain.entities.inference.events import QueryEvent


@dataclass(slots=True)
class TypingSegmentIngestService:
    """TypingSegmentPayload를 기존 inference pipeline에 연결한다."""

    pipeline_service: InferencePipelineService

    def process(
        self,
        segment: TypingSegmentPayload,
    ) -> TypingSegmentIngestResponsePayload:
        """단일 typing segment를 QueryEvent로 정규화해 처리한다."""

        result = self.pipeline_service.process(_segment_to_query_event(segment))
        top_category, top_score = _top_score(result)
        return TypingSegmentIngestResponsePayload(
            segment_id=segment.segment_id,
            query_id=result.analysis_event.query_id,
            top_category=top_category,
            top_score=top_score,
            message="typing segment가 처리되어 저장되었습니다.",
        )

    def process_batch(
        self,
        segments: tuple[TypingSegmentPayload, ...],
    ) -> TypingSegmentBatchIngestResponsePayload:
        """여러 typing segment를 순서대로 처리한다."""

        results = tuple(self.process(segment) for segment in segments)
        return TypingSegmentBatchIngestResponsePayload(
            processed=len(results),
            results=results,
        )


def _segment_to_query_event(segment: TypingSegmentPayload) -> QueryEvent:
    source_type = f"{segment.source_type.value}:{segment.surface_type.value}"
    return QueryEvent(
        query_id=segment.segment_id,
        text=segment.analysis_text,
        occurred_at=segment.ended_at,
        locale=segment.locale,
        source_type=source_type,
    )


def _top_score(
    result: InferencePipelineResult,
) -> tuple[str | None, float | None]:
    scores = result.analysis_event.category_scores
    if not scores:
        return None, None
    top_category = max(scores, key=scores.__getitem__)
    return top_category, scores[top_category]

