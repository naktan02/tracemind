"""Captured text view generation + analysis debug job 실행 서비스."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.contracts.captured_text_contracts import (
    CapturedTextDebugJobRunResultPayload,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextAnalysisSourceRecord,
    CapturedTextRepository,
)
from agent.src.services.captured_text.lifecycle import (
    CapturedTextLifecycleService,
)
from agent.src.services.captured_text.view_generation.service import (
    CapturedTextViewGenerationService,
)
from agent.src.services.inference.pipeline_service import InferencePipelineService
from shared.src.domain.entities.inference.events import QueryEvent


@dataclass(slots=True)
class CapturedTextDebugJobService:
    """Pending captured text의 view generation과 weak text 분석을 한 번 실행한다."""

    repository: CapturedTextRepository
    view_generation_service: CapturedTextViewGenerationService
    lifecycle_service: CapturedTextLifecycleService
    pipeline_service: InferencePipelineService | None = None

    def run_once(self, *, limit: int) -> CapturedTextDebugJobRunResultPayload:
        """pending raw event를 view로 만든 뒤 analysis pending row를 분류한다."""

        self.lifecycle_service.purge(repository=self.repository)
        view_result = self.view_generation_service.generate_pending_views(limit=limit)
        return self._run_pending_analysis(view_result=view_result, limit=limit)

    def _run_pending_analysis(
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

        sources = self.repository.get_pending_analysis_sources(limit=limit)
        processed_count = 0
        failed_count = 0
        last_error = ""
        for source in sources:
            try:
                result = self.pipeline_service.process(_source_to_query_event(source))
                analysis_id = result.analysis_event.analysis_id or source.event_id
                self.repository.mark_analysis_completed(
                    event_id=source.event_id,
                    analysis_id=analysis_id,
                )
            except Exception as exc:
                failed_count += 1
                last_error = str(exc)
                self.repository.mark_analysis_failed(
                    event_id=source.event_id,
                    error_message=last_error,
                )
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
                "analysis_selected_count": len(sources),
                "analysis_processed_count": processed_count,
                "analysis_failed_count": failed_count,
                "message": message,
            }
        )


def _source_to_query_event(source: CapturedTextAnalysisSourceRecord) -> QueryEvent:
    weak_locale = source.metadata.get("weak_text_target_locale")
    return QueryEvent(
        query_id=source.event_id,
        text=source.weak_text,
        occurred_at=source.occurred_at,
        locale=weak_locale if isinstance(weak_locale, str) else source.locale,
        source_type=f"captured_text_weak:{source.source_type}:{source.surface_type}",
    )


def _append_run_message(current: str, addition: str) -> str:
    if not current:
        return addition
    return f"{current} {addition}"
