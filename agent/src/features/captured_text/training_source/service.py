"""Captured text generated view를 training source row로 투영한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from agent.src.features.captured_text.storage.records import (
    CapturedTextGeneratedTrainingSourceRecord,
)
from agent.src.features.captured_text.storage.repository import (
    CapturedTextRepository,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(slots=True)
class CapturedTextTrainingSourceService:
    """agent-local generated view를 학습 backend 입력 shape로 정규화한다."""

    repository: CapturedTextRepository

    def get_recent_query_ssl_unlabeled_rows(
        self,
        *,
        days: int,
        limit: int,
        as_of: datetime | None = None,
    ) -> tuple[LabeledQueryRow, ...]:
        """최근 ready generated view를 Query SSL unlabeled row로 반환한다."""

        if days <= 0:
            raise ValueError("days must be positive.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        effective_as_of = as_of or datetime.now(tz=timezone.utc)
        records = self.repository.get_ready_generated_training_sources(
            cutoff=effective_as_of - timedelta(days=days),
            limit=limit,
        )
        return tuple(_to_query_ssl_unlabeled_row(record) for record in records)


def _to_query_ssl_unlabeled_row(
    record: CapturedTextGeneratedTrainingSourceRecord,
) -> LabeledQueryRow:
    weak_text_was_translated = record.metadata.get("weak_text_translated") is True
    return {
        "query_id": record.event_id,
        "text": record.weak_text,
        "raw_label_scheme": "agent_unlabeled",
        "raw_label": "",
        "mapped_label_4": "",
        "locale": "en" if weak_text_was_translated else record.locale,
        "annotation_source": "agent_local_unlabeled",
        "approved_by": None,
        "created_at": record.generated_at.isoformat(),
        "aug_0": record.strong_text_0,
        "aug_1": record.strong_text_1,
        "weak_text": record.weak_text,
        "strong_text": record.strong_text_0,
        "weak_translated_text": record.weak_text if weak_text_was_translated else "",
        "strong_translated_text": (
            record.strong_text_0 if weak_text_was_translated else ""
        ),
    }
