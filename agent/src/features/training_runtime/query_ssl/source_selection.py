"""Query SSL live task source row selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent.src.features.captured_text.storage.repository import (
    CapturedTextRepository,
)
from agent.src.features.captured_text.training_source.service import (
    CapturedTextTrainingSourceService,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
    StoredAnalysisEvent,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(frozen=True, slots=True)
class QuerySslTrainingSourceSelection:
    """Query SSL local update에 사용할 labeled/unlabeled row 묶음."""

    labels: tuple[str, ...]
    labeled_rows: tuple[LabeledQueryRow, ...]
    unlabeled_rows: tuple[LabeledQueryRow, ...]


def select_query_ssl_training_sources(
    *,
    analysis_event_repository: AnalysisEventRepository,
    captured_text_repository: CapturedTextRepository,
    analysis_event_days: int,
    labels: Sequence[str],
    max_unlabeled_rows: int,
) -> QuerySslTrainingSourceSelection:
    """agent-local 저장소에서 Query SSL labeled/unlabeled source를 고른다."""

    normalized_labels = tuple(str(label) for label in labels)
    labeled_rows = build_labeled_rows_from_stored_events(
        stored_events=analysis_event_repository.get_recent_stored(
            days=analysis_event_days
        ),
        labels=normalized_labels,
    )
    unlabeled_rows = CapturedTextTrainingSourceService(
        repository=captured_text_repository
    ).get_recent_query_ssl_unlabeled_rows(
        days=analysis_event_days,
        limit=max_unlabeled_rows,
    )
    return QuerySslTrainingSourceSelection(
        labels=normalized_labels,
        labeled_rows=labeled_rows,
        unlabeled_rows=tuple(unlabeled_rows),
    )


def build_labeled_rows_from_stored_events(
    *,
    stored_events: Sequence[StoredAnalysisEvent],
    labels: Sequence[str],
) -> tuple[LabeledQueryRow, ...]:
    """stored analysis event를 Query SSL labeled anchor row로 정규화한다."""

    label_set = {str(label) for label in labels}
    rows: list[LabeledQueryRow] = []
    for stored_event in stored_events:
        event = stored_event.analysis_event
        if event.translated_text is None or not event.category_scores:
            continue
        label = _top_label(event.category_scores)
        if label not in label_set:
            continue
        rows.append(
            {
                "query_id": event.query_id,
                "text": event.translated_text,
                "raw_label_scheme": "agent_local_pseudo_label",
                "raw_label": label,
                "mapped_label_4": label,
                "locale": "en",
                "annotation_source": "agent_local_analysis_event",
                "approved_by": None,
                "created_at": event.occurred_at.isoformat(),
            }
        )
    return tuple(rows)


def _top_label(category_scores: Mapping[str, float]) -> str:
    return max(category_scores.items(), key=lambda item: float(item[1]))[0]
