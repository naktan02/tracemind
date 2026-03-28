"""WindowSummaryBuilder unit tests."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PROJECT_SHARED_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SHARED_ROOT))

from src.domain.entities.scored_event import ScoredEvent
from src.domain.services.window_summary_builder import WindowSummaryBuilder


def test_build_aggregates_scored_events_into_window_summary() -> None:
    builder = WindowSummaryBuilder()
    base_time = datetime(2026, 3, 28, 9, 0, 0)
    scored_events = [
        ScoredEvent(
            query_id="q1",
            occurred_at=base_time,
            translated_text="sleep help",
            embedding_model_id="mxbai",
            translation_model_id=None,
            category_scores={"anxiety": 0.8, "normal": 0.2},
        ),
        ScoredEvent(
            query_id="q2",
            occurred_at=base_time + timedelta(minutes=5),
            translated_text="stress symptoms",
            embedding_model_id="mxbai",
            translation_model_id=None,
            category_scores={"anxiety": 0.4, "depression": 0.6, "normal": 0.1},
        ),
    ]

    summary = builder.build(
        scored_events,
        age_band="13_15",
        summary_id="summary-test-001",
    )

    assert summary.schema_version == "window_summary.v1"
    assert summary.summary_id == "summary-test-001"
    assert summary.age_band == "13_15"
    assert summary.batch_started_at == base_time
    assert summary.batch_ended_at == base_time + timedelta(minutes=5)
    assert summary.event_count == 2
    assert summary.category_stats["anxiety"].mean == pytest.approx(0.6)
    assert summary.category_stats["anxiety"].max == pytest.approx(0.8)
    assert summary.category_stats["anxiety"].count == 2
    assert summary.category_stats["depression"].mean == pytest.approx(0.6)
    assert summary.category_stats["depression"].max == pytest.approx(0.6)
    assert summary.category_stats["depression"].count == 1
    assert summary.category_stats["normal"].mean == pytest.approx(0.15)
    assert summary.category_stats["normal"].max == pytest.approx(0.2)
    assert summary.category_stats["normal"].count == 2


def test_build_rejects_empty_age_band() -> None:
    builder = WindowSummaryBuilder()

    with pytest.raises(ValueError, match="age_band must not be empty"):
        builder.build(
            [
                ScoredEvent(
                    query_id="q1",
                    occurred_at=datetime(2026, 3, 28, 9, 0, 0),
                    translated_text=None,
                    embedding_model_id="mxbai",
                    translation_model_id=None,
                    category_scores={"anxiety": 0.5},
                )
            ],
            age_band="   ",
        )


def test_build_rejects_empty_scored_events() -> None:
    builder = WindowSummaryBuilder()

    with pytest.raises(ValueError, match="scored_events must not be empty"):
        builder.build([], age_band="13_15")


def test_build_rejects_missing_category_scores() -> None:
    builder = WindowSummaryBuilder()

    with pytest.raises(
        ValueError,
        match="At least one category score is required to build a summary",
    ):
        builder.build(
            [
                ScoredEvent(
                    query_id="q1",
                    occurred_at=datetime(2026, 3, 28, 9, 0, 0),
                    translated_text=None,
                    embedding_model_id="mxbai",
                    translation_model_id=None,
                    category_scores={},
                )
            ],
            age_band="13_15",
        )
