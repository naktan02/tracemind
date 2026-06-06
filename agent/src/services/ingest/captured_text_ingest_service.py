"""Captured text event를 agent-local inference 입력으로 변환하는 service."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
    captured_text_record_from_payload,
)
from agent.src.services.inference.pipeline_service import (
    InferencePipelineResult,
    InferencePipelineService,
)
from shared.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestResponsePayload,
    CapturedTextEventPayload,
    CapturedTextIngestResponsePayload,
)
from shared.src.domain.entities.inference.events import QueryEvent


@dataclass(slots=True)
class CapturedTextIngestService:
    """CapturedTextEventPayload를 저장하고 기존 inference pipeline에 연결한다."""

    pipeline_service: InferencePipelineService
    captured_text_repository: CapturedTextRepository

    def process(
        self,
        event: CapturedTextEventPayload,
    ) -> CapturedTextIngestResponsePayload:
        """단일 captured text event를 저장한 뒤 QueryEvent로 정규화해 처리한다."""

        self.captured_text_repository.save(captured_text_record_from_payload(event))
        result = self.pipeline_service.process(_captured_text_to_query_event(event))
        top_category, top_score = _top_score(result)
        return CapturedTextIngestResponsePayload(
            event_id=event.event_id,
            query_id=result.scored_event.query_id,
            top_category=top_category,
            top_score=top_score,
            message="captured text event가 처리되어 저장되었습니다.",
        )

    def process_batch(
        self,
        events: tuple[CapturedTextEventPayload, ...],
    ) -> CapturedTextBatchIngestResponsePayload:
        """여러 captured text event를 순서대로 처리한다."""

        results = tuple(self.process(event) for event in events)
        return CapturedTextBatchIngestResponsePayload(
            processed=len(results),
            results=results,
        )


def _captured_text_to_query_event(event: CapturedTextEventPayload) -> QueryEvent:
    return QueryEvent(
        query_id=event.event_id,
        text=event.text,
        occurred_at=event.occurred_at,
        locale=event.locale,
        source_type=_query_source_type(event),
    )


def _query_source_type(event: CapturedTextEventPayload) -> str:
    return f"{event.source_type.value}:{event.surface_type.value}"


def _top_score(
    result: InferencePipelineResult,
) -> tuple[str | None, float | None]:
    scores = result.scored_event.category_scores
    if not scores:
        return None, None
    top_category = max(scores, key=scores.__getitem__)
    return top_category, scores[top_category]
