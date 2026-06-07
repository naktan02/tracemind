"""Captured text debug job 실행 서비스."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.contracts.captured_text_contracts import (
    CapturedTextDebugJobRunResultPayload,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextRecord,
    CapturedTextRepository,
)
from agent.src.services.inference.pipeline_service import InferencePipelineService
from agent.src.services.ingest.captured_text_lifecycle_service import (
    CapturedTextLifecycleService,
)
from agent.src.services.ingest.captured_text_view_generation_service import (
    CapturedTextViewGenerationService,
)
from shared.src.domain.entities.inference.events import QueryEvent


@dataclass(slots=True)
class CapturedTextDebugJobService:
    """View generation과 미분석 ready event 분석을 한 번 실행한다."""

    repository: CapturedTextRepository
    view_generation_service: CapturedTextViewGenerationService
    lifecycle_service: CapturedTextLifecycleService
    pipeline_service: InferencePipelineService | None = None

    def run_once(self, *, limit: int) -> CapturedTextDebugJobRunResultPayload:
        """pending view 생성 후, 아직 분석되지 않은 ready event를 처리한다."""

        self.lifecycle_service.purge(repository=self.repository)
        view_result = self.view_generation_service.generate_pending_views(limit=limit)
        return self._run_ready_analysis(view_result=view_result, limit=limit)

    def _run_ready_analysis(
        self,
        *,
        view_result: CapturedTextDebugJobRunResultPayload,
        limit: int,
    ) -> CapturedTextDebugJobRunResultPayload:
        if self.pipeline_service is None:
            return view_result.model_copy(
                update={
                    "message": _append_run_message(
                        view_result.message,
                        "analysis skipped: pipeline_service가 설정되지 않았습니다.",
                    )
                }
            )

        candidates = _ready_unanalysed_records(
            repository=self.repository,
            pipeline_service=self.pipeline_service,
            limit=limit,
        )
        processed_count = 0
        failed_count = 0
        last_error = ""
        for record in candidates:
            try:
                self.pipeline_service.process(_captured_record_to_query_event(record))
            except Exception as exc:
                failed_count += 1
                last_error = str(exc)
                continue
            processed_count += 1

        message = view_result.message
        if failed_count:
            message = _append_run_message(
                message,
                f"analysis failed: {failed_count}건. last_error={last_error}",
            )
        return view_result.model_copy(
            update={
                "analysis_selected_count": len(candidates),
                "analysis_processed_count": processed_count,
                "analysis_failed_count": failed_count,
                "message": message,
            }
        )


def _ready_unanalysed_records(
    *,
    repository: CapturedTextRepository,
    pipeline_service: InferencePipelineService,
    limit: int,
) -> list[CapturedTextRecord]:
    records = []
    for record in repository.get_recent(limit=limit):
        if record.view_generation_status != CAPTURED_TEXT_VIEW_STATUS_READY:
            continue
        if record.duplicate_of_event_id is not None:
            continue
        if repository.get_generated_view(record.event_id) is None:
            continue
        if pipeline_service.event_repository.has_source_event_id(record.event_id):
            continue
        records.append(record)
    return list(reversed(records))


def _captured_record_to_query_event(record: CapturedTextRecord) -> QueryEvent:
    return QueryEvent(
        query_id=record.event_id,
        text=record.text,
        occurred_at=record.occurred_at,
        locale=record.locale,
        source_type=f"{record.source_type}:{record.surface_type}",
    )


def _append_run_message(current: str, addition: str) -> str:
    if not current:
        return addition
    return f"{current} {addition}"
