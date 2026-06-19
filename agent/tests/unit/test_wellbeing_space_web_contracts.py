"""Wellbeing space-web contract unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent.src.contracts.wellbeing_signal_contracts import (
    WellbeingSignalLevel,
    WellbeingSignalRange,
    WellbeingSignalTrend,
)
from agent.src.contracts.wellbeing_space_web_contracts import (
    WellbeingSpaceWebEdgePayload,
    WellbeingSpaceWebNodePayload,
    WellbeingSpaceWebPayload,
    WellbeingSpaceWebRelationType,
    dump_wellbeing_space_web_payload,
    load_wellbeing_space_web_payload,
)


def test_wellbeing_space_web_payload_accepts_nodes_and_edges() -> None:
    payload = WellbeingSpaceWebPayload(
        computed_at=datetime(2026, 6, 13, 9, tzinfo=timezone.utc),
        range=WellbeingSignalRange.LAST_7_DAYS,
        strategy_name="coactivation_delta",
        strategy_version="v1",
        nodes=(
            WellbeingSpaceWebNodePayload(
                id="stress_signal",
                label="stress signal",
                intensity=72.5,
                level=WellbeingSignalLevel.HIGH,
                trend=WellbeingSignalTrend.RISING,
                observed_event_count=4,
            ),
            WellbeingSpaceWebNodePayload(
                id="isolation_signal",
                label="isolation signal",
                intensity=61.0,
                level=WellbeingSignalLevel.HIGH,
                trend=WellbeingSignalTrend.STEADY,
                observed_event_count=4,
            ),
        ),
        edges=(
            WellbeingSpaceWebEdgePayload(
                source="stress_signal",
                target="isolation_signal",
                weight=48.0,
                relation_type=WellbeingSpaceWebRelationType.COACTIVATION,
                evidence_count=3,
            ),
        ),
        low_data=False,
    )

    assert payload.schema_version == "wellbeing_space_web.v1"
    assert payload.nodes[0].level == WellbeingSignalLevel.HIGH
    assert payload.edges[0].relation_type == WellbeingSpaceWebRelationType.COACTIVATION


def test_wellbeing_space_web_payload_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        WellbeingSpaceWebPayload(
            computed_at=datetime(2026, 6, 13, 9, tzinfo=timezone.utc),
            range=WellbeingSignalRange.LAST_7_DAYS,
            strategy_name="coactivation_delta",
            strategy_version="v1",
            nodes=(
                WellbeingSpaceWebNodePayload(
                    id="stress_signal",
                    label="stress signal",
                    intensity=120.0,
                    level=WellbeingSignalLevel.VERY_HIGH,
                    trend=WellbeingSignalTrend.RISING,
                    observed_event_count=1,
                ),
            ),
        )


def test_wellbeing_space_web_dump_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "space_web.json"
    payload = WellbeingSpaceWebPayload(
        computed_at=datetime(2026, 6, 13, 9, tzinfo=timezone.utc),
        range=WellbeingSignalRange.LAST_14_DAYS,
        strategy_name="coactivation_delta",
        strategy_version="v1",
        nodes=(),
        edges=(),
        low_data=True,
    )

    dump_wellbeing_space_web_payload(path, payload)
    loaded = load_wellbeing_space_web_payload(path)

    assert loaded == payload
