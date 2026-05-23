"""Wellbeing projection service tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.scored_event_repository import (
    ScoredEventRepository,
)
from agent.src.infrastructure.repositories.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.services.wellbeing.projection_service import (
    WellbeingSignalProjectionService,
)
from agent.src.services.wellbeing.summary_service import WellbeingSummaryService
from agent.src.services.wellbeing.timeseries_service import WellbeingTimeseriesService
from shared.src.contracts.wellbeing_signal_contracts import WellbeingSignalRange
from shared.src.domain.entities.inference.events import ScoredEvent


def _build_scored_event(
    *,
    query_id: str,
    occurred_at: datetime,
    score: float,
) -> ScoredEvent:
    return ScoredEvent(
        query_id=query_id,
        occurred_at=occurred_at,
        translated_text=None,
        embedding_model_id="test-embedding",
        translation_model_id=None,
        category_scores={"stress_signal": score},
    )


def test_projection_service_replays_scored_events_into_snapshots(
    tmp_path: Path,
) -> None:
    scored_event_repository = ScoredEventRepository(
        db_path=tmp_path / "scored_events.db"
    )
    snapshot_repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    projection_service = WellbeingSignalProjectionService(
        scored_event_repository=scored_event_repository,
        snapshot_repository=snapshot_repository,
        lookback_days=60,
    )

    scored_event_repository.save(
        _build_scored_event(
            query_id="q1",
            occurred_at=datetime(2026, 4, 21, 9, tzinfo=timezone.utc),
            score=0.42,
        )
    )
    scored_event_repository.save(
        _build_scored_event(
            query_id="q2",
            occurred_at=datetime(2026, 4, 22, 9, tzinfo=timezone.utc),
            score=0.48,
        )
    )
    scored_event_repository.save(
        _build_scored_event(
            query_id="q3",
            occurred_at=datetime(2026, 4, 23, 9, tzinfo=timezone.utc),
            score=0.86,
        )
    )

    projection_service.refresh_from_runtime()

    latest_summary = snapshot_repository.load_latest_summary()
    assert latest_summary is not None
    assert latest_summary.computed_at == datetime(2026, 4, 23, 9, tzinfo=timezone.utc)
    assert snapshot_repository.list_summaries_since(
        cutoff=datetime(2026, 4, 20, tzinfo=timezone.utc)
    )
    assert (
        len(
            snapshot_repository.list_summaries_since(
                cutoff=datetime(2026, 4, 20, tzinfo=timezone.utc)
            )
        )
        == 3
    )


def test_wellbeing_services_refresh_projection_before_read(
    tmp_path: Path,
) -> None:
    scored_event_repository = ScoredEventRepository(
        db_path=tmp_path / "scored_events.db"
    )
    snapshot_repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    projection_service = WellbeingSignalProjectionService(
        scored_event_repository=scored_event_repository,
        snapshot_repository=snapshot_repository,
    )
    summary_service = WellbeingSummaryService(
        repository=snapshot_repository,
        projection_service=projection_service,
    )
    timeseries_service = WellbeingTimeseriesService(
        repository=snapshot_repository,
        projection_service=projection_service,
    )

    scored_event_repository.save(
        _build_scored_event(
            query_id="q1",
            occurred_at=datetime(2026, 4, 24, 9, tzinfo=timezone.utc),
            score=0.73,
        )
    )

    summary = summary_service.get_current_summary()
    timeseries = timeseries_service.get_timeseries(
        requested_range=WellbeingSignalRange.LAST_7_DAYS
    )

    assert summary.computed_at == datetime(2026, 4, 24, 9, tzinfo=timezone.utc)
    assert len(timeseries.points) == 1
    assert timeseries.points[0].ts == datetime(2026, 4, 24, 9, tzinfo=timezone.utc)
