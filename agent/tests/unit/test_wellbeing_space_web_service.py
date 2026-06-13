"""Wellbeing space-web projection tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalRange,
    WellbeingSignalTrend,
)
from agent.src.contracts.wellbeing_space_web_contracts import (
    WellbeingSpaceWebRelationType,
)
from agent.src.features.inference.interpretation.state import BaselineProfile
from agent.src.features.wellbeing.space_web.coactivation_delta_strategy import (
    CoactivationDeltaSpaceWebStrategy,
)
from agent.src.features.wellbeing.space_web.projection_service import (
    WellbeingSpaceWebProjectionService,
)
from agent.src.features.wellbeing.space_web.projection_strategy import (
    SpaceWebProjectionContext,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from shared.src.domain.entities.inference.events import AnalysisEvent


def test_coactivation_delta_strategy_builds_nodes_and_edges() -> None:
    computed_at = datetime(2026, 6, 13, 9, tzinfo=timezone.utc)
    strategy = CoactivationDeltaSpaceWebStrategy()

    result = strategy.project(
        SpaceWebProjectionContext(
            recent_events=(
                _analysis_event(
                    query_id="q1",
                    occurred_at=computed_at - timedelta(days=1),
                    scores={"stress_signal": 0.58, "isolation_signal": 0.52},
                ),
                _analysis_event(
                    query_id="q2",
                    occurred_at=computed_at,
                    scores={"stress_signal": 0.78, "isolation_signal": 0.72},
                ),
            ),
            baseline_profile=BaselineProfile(
                profile_version="baseline_profile.v1",
                warmup_complete=True,
                category_means={"stress_signal": 0.2, "isolation_signal": 0.18},
            ),
            computed_at=computed_at,
            category_label_overrides={},
        )
    )

    assert [node.id for node in result.nodes] == [
        "isolation_signal",
        "stress_signal",
    ]
    assert result.nodes[0].trend == WellbeingSignalTrend.RISING
    assert len(result.edges) == 1
    assert result.edges[0].relation_type == WellbeingSpaceWebRelationType.COACTIVATION
    assert result.edges[0].weight > 0


def test_coactivation_delta_strategy_keeps_normal_category_node() -> None:
    computed_at = datetime(2026, 6, 13, 9, tzinfo=timezone.utc)
    strategy = CoactivationDeltaSpaceWebStrategy()

    result = strategy.project(
        SpaceWebProjectionContext(
            recent_events=(
                _analysis_event(
                    query_id="q1",
                    occurred_at=computed_at,
                    scores={
                        "anxiety": 0.7,
                        "depression": 0.4,
                        "normal": 0.2,
                        "suicidal": 0.1,
                    },
                ),
            ),
            baseline_profile=BaselineProfile(
                profile_version="baseline_profile.v1",
                warmup_complete=True,
                category_means={
                    "anxiety": 0.1,
                    "depression": 0.1,
                    "normal": 0.3,
                    "suicidal": 0.05,
                },
            ),
            computed_at=computed_at,
            category_label_overrides={},
        )
    )

    assert {node.id for node in result.nodes} == {
        "anxiety",
        "depression",
        "normal",
        "suicidal",
    }


def test_space_web_service_returns_payload_from_recent_analysis_events(
    tmp_path: Path,
) -> None:
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    repository = AnalysisEventRepository(db_path=tmp_path / "analysis_events.db")
    repository.save(
        _analysis_event(
            query_id="baseline",
            occurred_at=now - timedelta(days=12),
            scores={"stress_signal": 0.2, "isolation_signal": 0.2},
        )
    )
    repository.save(
        _analysis_event(
            query_id="recent-1",
            occurred_at=now - timedelta(days=1),
            scores={"stress_signal": 0.7, "isolation_signal": 0.68},
        )
    )
    repository.save(
        _analysis_event(
            query_id="recent-2",
            occurred_at=now,
            scores={"stress_signal": 0.74, "isolation_signal": 0.71},
        )
    )
    service = WellbeingSpaceWebProjectionService(
        analysis_event_repository=repository,
        now_provider=lambda: now,
    )

    payload = service.get_space_web(requested_range=WellbeingSignalRange.LAST_7_DAYS)

    assert payload.range == WellbeingSignalRange.LAST_7_DAYS
    assert payload.strategy_name == "coactivation_delta"
    assert {node.id for node in payload.nodes} == {
        "stress_signal",
        "isolation_signal",
    }
    assert len(payload.edges) == 1
    assert payload.edges[0].weight > 0


def _analysis_event(
    *,
    query_id: str,
    occurred_at: datetime,
    scores: dict[str, float],
) -> AnalysisEvent:
    return AnalysisEvent(
        query_id=query_id,
        occurred_at=occurred_at,
        translated_text=None,
        embedding_model_id="test-embedding",
        translation_model_id=None,
        category_scores=scores,
    )
