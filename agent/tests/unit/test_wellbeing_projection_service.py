"""Wellbeing projection service tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalRange,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)
from agent.src.features.captured_text.storage.records import CapturedTextRecord
from agent.src.features.captured_text.storage.repository import CapturedTextRepository
from agent.src.features.wellbeing.range_window import cutoff_for_range
from agent.src.features.wellbeing.signal.projection_service import (
    WellbeingSignalProjectionService,
)
from agent.src.features.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.features.wellbeing.signal.timeseries_service import (
    WellbeingTimeseriesService,
)
from agent.src.features.wellbeing.storage.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from shared.src.domain.entities.inference.events import AnalysisEvent


def test_wellbeing_range_cutoff_uses_last_24_hours_for_one_day() -> None:
    anchor = datetime(2026, 6, 13, 9, tzinfo=timezone.utc)

    assert cutoff_for_range(anchor, WellbeingSignalRange.LAST_1_DAY) == anchor - (
        timedelta(days=1)
    )
    assert cutoff_for_range(anchor, WellbeingSignalRange.LAST_7_DAYS) == anchor - (
        timedelta(days=6)
    )


def _build_analysis_event(
    *,
    query_id: str,
    occurred_at: datetime,
    score: float,
) -> AnalysisEvent:
    return AnalysisEvent(
        query_id=query_id,
        occurred_at=occurred_at,
        translated_text=None,
        embedding_model_id="test-embedding",
        translation_model_id=None,
        category_scores={"stress_signal": score},
    )


def test_projection_service_replays_analysis_events_into_snapshots(
    tmp_path: Path,
) -> None:
    analysis_event_repository = AnalysisEventRepository(
        db_path=tmp_path / "analysis_events.db"
    )
    snapshot_repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    projection_service = WellbeingSignalProjectionService(
        analysis_event_repository=analysis_event_repository,
        snapshot_repository=snapshot_repository,
        lookback_days=60,
    )

    analysis_event_repository.save(
        _build_analysis_event(
            query_id="q1",
            occurred_at=datetime(2026, 4, 21, 9, tzinfo=timezone.utc),
            score=0.42,
        )
    )
    analysis_event_repository.save(
        _build_analysis_event(
            query_id="q2",
            occurred_at=datetime(2026, 4, 22, 9, tzinfo=timezone.utc),
            score=0.48,
        )
    )
    analysis_event_repository.save(
        _build_analysis_event(
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


def test_projection_service_skips_replay_when_snapshots_are_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_event_repository = AnalysisEventRepository(
        db_path=tmp_path / "analysis_events.db"
    )
    snapshot_repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    projection_service = WellbeingSignalProjectionService(
        analysis_event_repository=analysis_event_repository,
        snapshot_repository=snapshot_repository,
        lookback_days=60,
    )
    analysis_event_repository.save(
        _build_analysis_event(
            query_id="q1",
            occurred_at=datetime(2026, 4, 24, 9, tzinfo=timezone.utc),
            score=0.73,
        )
    )
    projection_service.refresh_from_runtime()

    def fail_get_recent(*args, **kwargs):
        raise AssertionError("current snapshots should not replay analysis events")

    monkeypatch.setattr(AnalysisEventRepository, "get_recent", fail_get_recent)

    projection_service.refresh_from_runtime()


def test_projection_service_replays_when_projection_version_is_legacy(
    tmp_path: Path,
) -> None:
    analysis_event_repository = AnalysisEventRepository(
        db_path=tmp_path / "analysis_events.db"
    )
    snapshot_repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    projection_service = WellbeingSignalProjectionService(
        analysis_event_repository=analysis_event_repository,
        snapshot_repository=snapshot_repository,
        lookback_days=60,
    )
    occurred_at = datetime(2026, 4, 24, 9, tzinfo=timezone.utc)
    analysis_event_repository.save(
        _build_analysis_event(query_id="q1", occurred_at=occurred_at, score=0.73)
    )
    snapshot_repository.save_summary(
        WellbeingSignalSummaryPayload(
            computed_at=occurred_at,
            signal_score=1.0,
            signal_level=WellbeingSignalLevel.LOW,
            signal_label="안정",
            trend=WellbeingSignalTrend.UNKNOWN,
            summary="legacy snapshot",
            action_tip="legacy tip",
            confidence=WellbeingSignalConfidence.LOW,
            low_data=True,
        )
    )

    projection_service.refresh_from_runtime()

    latest_summary = snapshot_repository.load_latest_summary()
    assert latest_summary is not None
    assert latest_summary.summary != "legacy snapshot"
    assert (
        snapshot_repository.load_latest_projection_version()
        == "wellbeing_projection.evidence_signal.v1"
    )


def test_projection_service_uses_direct_risk_source_text_as_evidence(
    tmp_path: Path,
) -> None:
    analysis_event_repository = AnalysisEventRepository(
        db_path=tmp_path / "analysis_events.db"
    )
    captured_text_repository = CapturedTextRepository(
        db_path=tmp_path / "analysis_events.db"
    )
    snapshot_repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    projection_service = WellbeingSignalProjectionService(
        analysis_event_repository=analysis_event_repository,
        snapshot_repository=snapshot_repository,
        captured_text_repository=captured_text_repository,
        lookback_days=60,
    )
    occurred_at = datetime(2026, 6, 14, 9, tzinfo=timezone.utc)
    captured_text_repository.save(
        CapturedTextRecord(
            event_id="captured-risk-1",
            occurred_at=occurred_at,
            received_at=occurred_at,
            text="자살 어떻게 할 수 있지",
            locale="ko",
            source_type="browser",
            surface_type="rich_editor",
        )
    )
    analysis_event_repository.save(
        AnalysisEvent(
            query_id="captured-risk-1",
            occurred_at=occurred_at,
            translated_text="How can I commit suicide?",
            embedding_model_id="test-embedding",
            translation_model_id=None,
            category_scores={"normal": 0.80, "suicidal": 0.09},
        ),
        source_event_id="captured-risk-1",
        scorer_name="test-scorer",
        model_revision="test-revision",
    )

    projection_service.refresh_from_runtime()

    latest_summary = snapshot_repository.load_latest_summary()
    assert latest_summary is not None
    assert latest_summary.signal_level == WellbeingSignalLevel.VERY_HIGH
    assert latest_summary.signal_score == 95.0
    assert latest_summary.low_data is False


def test_wellbeing_services_refresh_projection_before_read(
    tmp_path: Path,
) -> None:
    analysis_event_repository = AnalysisEventRepository(
        db_path=tmp_path / "analysis_events.db"
    )
    snapshot_repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    projection_service = WellbeingSignalProjectionService(
        analysis_event_repository=analysis_event_repository,
        snapshot_repository=snapshot_repository,
        lookback_days=60,
    )
    summary_service = WellbeingSummaryService(
        repository=snapshot_repository,
        projection_service=projection_service,
    )
    timeseries_service = WellbeingTimeseriesService(
        repository=snapshot_repository,
        projection_service=projection_service,
    )

    analysis_event_repository.save(
        _build_analysis_event(
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
