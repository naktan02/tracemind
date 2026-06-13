"""Wellbeing storage and service integration tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent.src.config.paths import (
    DEFAULT_AGENT_DATA_DIR,
)
from agent.src.contracts.family_access_contracts import FamilyAccessRole
from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalConfidence,
    WellbeingSignalLevel,
    WellbeingSignalRange,
    WellbeingSignalSummaryPayload,
    WellbeingSignalTrend,
)
from agent.src.features.wellbeing.family_access.service import FamilyAccessService
from agent.src.features.wellbeing.signal.summary_service import WellbeingSummaryService
from agent.src.features.wellbeing.signal.timeseries_service import (
    WellbeingTimeseriesService,
)
from agent.src.features.wellbeing.storage.child_support_repository import (
    ChildSupportConversationRepository,
)
from agent.src.features.wellbeing.storage.family_access_repository import (
    FamilyAccessRepository,
    FamilyAccessState,
)
from agent.src.features.wellbeing.storage.wellbeing_settings_repository import (
    WellbeingSettingsRecord,
    WellbeingSettingsRepository,
)
from agent.src.features.wellbeing.storage.wellbeing_snapshot_repository import (
    WellbeingSnapshotRepository,
)
from agent.src.features.wellbeing.storage.wellbeing_storage import (
    DEFAULT_WELLBEING_DB_PATH,
)


def _build_summary_payload(
    *,
    computed_at: datetime,
    signal_score: float,
) -> WellbeingSignalSummaryPayload:
    return WellbeingSignalSummaryPayload(
        computed_at=computed_at,
        signal_score=signal_score,
        signal_level=WellbeingSignalLevel.MODERATE,
        signal_label="관찰 필요",
        trend=WellbeingSignalTrend.STEADY,
        summary="최근 상태가 비교적 안정적으로 유지되고 있습니다.",
        action_tip="오늘 저녁에 짧게 안부를 물어보세요.",
        confidence=WellbeingSignalConfidence.MEDIUM,
        low_data=False,
    )


def test_wellbeing_snapshot_repository_round_trips_latest_and_since(
    tmp_path: Path,
) -> None:
    repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    older = _build_summary_payload(
        computed_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        signal_score=44.0,
    )
    latest = _build_summary_payload(
        computed_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        signal_score=61.0,
    )

    repository.save_summary(older)
    repository.save_summary(latest)

    assert repository.load_latest_summary() == latest
    assert repository.load_latest_projection_version() == "legacy"
    assert repository.list_summaries_since(
        cutoff=datetime(2026, 4, 23, tzinfo=timezone.utc)
    ) == (latest,)


def test_wellbeing_default_storage_paths_use_agent_data_dir() -> None:
    child_support_repository = ChildSupportConversationRepository()
    wellbeing_repository = WellbeingSnapshotRepository()

    assert DEFAULT_WELLBEING_DB_PATH == DEFAULT_AGENT_DATA_DIR / "wellbeing_signal.db"
    assert child_support_repository.db_path == (
        DEFAULT_AGENT_DATA_DIR / "child_support_conversations.db"
    )
    assert wellbeing_repository.db_path == DEFAULT_WELLBEING_DB_PATH


def test_family_access_repository_round_trips_state(tmp_path: Path) -> None:
    repository = FamilyAccessRepository(db_path=tmp_path / "wellbeing.db")
    state = FamilyAccessState(
        role=FamilyAccessRole.PARENT,
        pin_hash="hash_parent_1234",
        failed_attempt_count=2,
        locked_until=datetime(2026, 4, 24, 10, 30, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 24, 10, 20, tzinfo=timezone.utc),
    )

    repository.save_state(state)

    assert repository.load_state(FamilyAccessRole.PARENT) == state


def test_wellbeing_settings_repository_load_or_default_then_persist(
    tmp_path: Path,
) -> None:
    repository = WellbeingSettingsRepository(db_path=tmp_path / "wellbeing.db")

    default_settings = repository.load_or_default()
    assert default_settings.default_timeseries_range == WellbeingSignalRange.LAST_7_DAYS

    updated_settings = WellbeingSettingsRecord(
        default_timeseries_range=WellbeingSignalRange.LAST_30_DAYS,
        child_session_minutes=8,
        child_lock_minutes=2,
        child_max_attempts=4,
        parent_session_minutes=20,
        parent_lock_minutes=15,
        parent_max_attempts=7,
    )
    repository.save_settings(updated_settings)

    assert repository.load_settings() == updated_settings


def test_wellbeing_services_prefer_repository_data(tmp_path: Path) -> None:
    repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")
    summary_payload = _build_summary_payload(
        computed_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        signal_score=58.0,
    )
    repository.save_summary(summary_payload)

    summary_service = WellbeingSummaryService(repository=repository)
    timeseries_service = WellbeingTimeseriesService(repository=repository)

    assert summary_service.get_current_summary() == summary_payload

    timeseries = timeseries_service.get_timeseries(
        requested_range=WellbeingSignalRange.LAST_7_DAYS
    )
    assert len(timeseries.points) == 1
    assert timeseries.points[0].signal_score == 58.0


def test_wellbeing_services_return_low_data_when_repository_is_empty(
    tmp_path: Path,
) -> None:
    repository = WellbeingSnapshotRepository(db_path=tmp_path / "wellbeing.db")

    summary = WellbeingSummaryService(repository=repository).get_current_summary()
    timeseries = WellbeingTimeseriesService(repository=repository).get_timeseries(
        requested_range=WellbeingSignalRange.LAST_7_DAYS
    )

    assert summary.low_data is True
    assert summary.signal_score == 0.0
    assert timeseries.points == ()


def test_family_access_service_persists_failed_attempts_in_repository(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wellbeing.db"
    auth_repository = FamilyAccessRepository(db_path=db_path)
    settings_repository = WellbeingSettingsRepository(db_path=db_path)
    settings_repository.save_settings(
        WellbeingSettingsRecord(
            child_max_attempts=3,
            child_lock_minutes=1,
            parent_max_attempts=3,
            parent_lock_minutes=1,
            parent_session_minutes=10,
        )
    )
    service = FamilyAccessService(
        repository=auth_repository,
        settings_repository=settings_repository,
    )
    service.create_initial_setup(child_pin="1111", parent_pin="1234")

    first_failure = service.unlock(role=FamilyAccessRole.PARENT, pin="9999")
    second_failure = service.unlock(role=FamilyAccessRole.PARENT, pin="9999")

    assert first_failure.granted is False
    assert second_failure.remaining_attempts == 1
    assert auth_repository.load_state(FamilyAccessRole.PARENT) is not None
    assert auth_repository.load_state(FamilyAccessRole.PARENT).failed_attempt_count == 2

    locked_response = service.unlock(role=FamilyAccessRole.PARENT, pin="9999")
    assert locked_response.remaining_attempts == 0
    assert locked_response.locked_until is not None
