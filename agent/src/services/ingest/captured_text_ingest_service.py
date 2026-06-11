"""Captured text event를 agent-local raw 입력으로 저장하는 service."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.contracts.captured_text_contracts import (
    CapturedTextBatchIngestResponsePayload,
    CapturedTextEventPayload,
    CapturedTextIngestResponsePayload,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
    captured_text_record_from_payload,
)
from agent.src.services.ingest.captured_text_lifecycle_service import (
    CapturedTextLifecycleService,
)


@dataclass(slots=True)
class CapturedTextIngestService:
    """CapturedTextEventPayload를 agent-local raw source of truth로 저장한다."""

    captured_text_repository: CapturedTextRepository
    lifecycle_service: CapturedTextLifecycleService | None = None

    def process(
        self,
        event: CapturedTextEventPayload,
    ) -> CapturedTextIngestResponsePayload:
        """단일 captured text event를 저장한다."""

        self._save_event(event)
        self._purge_if_configured()
        return _stored_response(event)

    def _purge_if_configured(self) -> None:
        if self.lifecycle_service is None:
            return
        self.lifecycle_service.purge(repository=self.captured_text_repository)

    def process_batch(
        self,
        events: tuple[CapturedTextEventPayload, ...],
    ) -> CapturedTextBatchIngestResponsePayload:
        """여러 captured text event를 저장한다."""

        for event in events:
            self._save_event(event)

        self._purge_if_configured()
        return CapturedTextBatchIngestResponsePayload(
            processed=len(events),
            results=tuple(_stored_response(event) for event in events),
        )

    def _save_event(self, event: CapturedTextEventPayload) -> None:
        self.captured_text_repository.save(captured_text_record_from_payload(event))


def _stored_response(
    event: CapturedTextEventPayload,
) -> CapturedTextIngestResponsePayload:
    return CapturedTextIngestResponsePayload(
        event_id=event.event_id,
        query_id=event.event_id,
        top_category=None,
        top_score=None,
        message="captured text event가 저장되었습니다.",
    )
