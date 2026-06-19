"""Local interpretation unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from agent.src.features.inference.interpretation.baseline import (
    BaselineConfig,
    BaselineService,
)
from agent.src.features.inference.interpretation.decision import DecisionService
from agent.src.features.inference.interpretation.decision_policy import (
    RuleBasedDecisionPolicy,
)
from agent.src.features.inference.interpretation.state import (
    BaselineProfile,
)
from agent.src.features.inference.interpretation.time_series import (
    TimeSeriesAccumulator,
    TimeSeriesConfig,
)
from shared.src.domain.entities.inference.events import AnalysisEvent


def _analysis_event(
    *,
    query_id: str,
    occurred_at: datetime,
    depression: float,
    normal: float = 0.2,
) -> AnalysisEvent:
    return AnalysisEvent(
        query_id=query_id,
        occurred_at=occurred_at,
        translated_text=None,
        embedding_model_id="mxbai",
        translation_model_id=None,
        category_scores={
            "depression": depression,
            "normal": normal,
        },
    )


def test_baseline_service_builds_profile_from_recent_history() -> None:
    base_time = datetime(2026, 3, 29, 9, 0, 0)
    history = [
        _analysis_event(
            query_id=f"q{i}",
            occurred_at=base_time - timedelta(days=3 - i),
            depression=value,
        )
        for i, value in enumerate([0.20, 0.25, 0.22, 0.24], start=0)
    ]
    service = BaselineService(
        config=BaselineConfig(
            lookback_days=14,
            warmup_days=3,
            warmup_events=4,
            minimum_sigma=0.01,
        )
    )

    profile = service.build_profile(history, as_of=base_time)

    assert profile.warmup_complete is True
    assert profile.event_count == 4
    assert profile.observed_days == 4
    assert profile.category_means["depression"] == 0.2275
    assert profile.category_latest["depression"] == 0.24
    assert profile.category_sigmas["depression"] > 0.01


def test_time_series_accumulator_tracks_elevated_streaks() -> None:
    base_time = datetime(2026, 3, 29, 12, 0, 0)
    profile = BaselineProfile(
        profile_version="baseline_profile.v1",
        warmup_complete=True,
        category_means={"depression": 0.2},
    )
    accumulator = TimeSeriesAccumulator(
        config=TimeSeriesConfig(ewma_alpha=0.5, default_delta_threshold=0.15)
    )

    first_state = accumulator.update(
        analysis_event=_analysis_event(
            query_id="q1",
            occurred_at=base_time,
            depression=0.5,
        ),
        baseline_profile=profile,
    )
    second_state = accumulator.update(
        analysis_event=_analysis_event(
            query_id="q2",
            occurred_at=base_time + timedelta(hours=1),
            depression=0.55,
        ),
        baseline_profile=profile,
        previous_state=first_state,
    )

    assert first_state.latest_deltas["depression"] == 0.3
    assert first_state.elevated_streaks["depression"] == 1
    assert second_state.elevated_streaks["depression"] == 2
    assert second_state.ewma_deltas["depression"] == 0.325


def test_decision_service_distinguishes_spike_from_persistent_change() -> None:
    base_time = datetime(2026, 3, 29, 15, 0, 0)
    profile = BaselineProfile(
        profile_version="baseline_profile.v1",
        warmup_complete=True,
        category_means={"depression": 0.2},
        category_sigmas={"depression": 0.05},
    )
    service = DecisionService(
        accumulator=TimeSeriesAccumulator(
            config=TimeSeriesConfig(ewma_alpha=0.5, default_delta_threshold=0.2)
        ),
        policy=RuleBasedDecisionPolicy(
            score_floor=0.5,
            default_delta_threshold=0.2,
            support_delta_multiplier=1.5,
            risk_delta_multiplier=2.0,
            watch_streak=1,
            support_streak=2,
            risk_streak=3,
        ),
    )

    first = service.evaluate(
        analysis_event=_analysis_event(
            query_id="q1",
            occurred_at=base_time,
            depression=0.55,
        ),
        baseline_profile=profile,
    )
    second = service.evaluate(
        analysis_event=_analysis_event(
            query_id="q2",
            occurred_at=base_time + timedelta(hours=1),
            depression=0.62,
        ),
        baseline_profile=profile,
        previous_state=first.time_series_state,
    )
    third = service.evaluate(
        analysis_event=_analysis_event(
            query_id="q3",
            occurred_at=base_time + timedelta(hours=2),
            depression=0.68,
        ),
        baseline_profile=profile,
        previous_state=second.time_series_state,
    )

    assert first.assessment_result.decision == "WATCH"
    assert first.assessment_result.focus_category == "depression"
    assert second.assessment_result.decision == "SUPPORT"
    assert third.assessment_result.decision == "RISK"
