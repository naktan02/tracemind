"""Captured text view generation debug job 실행 서비스."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.contracts.captured_text_contracts import (
    CapturedTextDebugJobRunResultPayload,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
)
from agent.src.services.ingest.captured_text_lifecycle_service import (
    CapturedTextLifecycleService,
)
from agent.src.services.ingest.captured_text_view_generation_service import (
    CapturedTextViewGenerationService,
)


@dataclass(slots=True)
class CapturedTextDebugJobService:
    """Pending captured text의 weak/strong view generation을 한 번 실행한다."""

    repository: CapturedTextRepository
    view_generation_service: CapturedTextViewGenerationService
    lifecycle_service: CapturedTextLifecycleService

    def run_once(self, *, limit: int) -> CapturedTextDebugJobRunResultPayload:
        """pending raw event를 weak/strong view로 materialize한다."""

        self.lifecycle_service.purge(repository=self.repository)
        return self.view_generation_service.generate_pending_views(limit=limit)
