"""Shared adapter aggregation backend tests."""

from __future__ import annotations

from datetime import datetime, timezone

from main_server.src.services.rounds.aggregation_service import (
    DiagonalScaleAggregationService,
    build_shared_adapter_aggregation_backend,
)
from main_server.src.services.rounds.diagonal_scale_defaults import (
    DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG,
)
from shared.src.contracts.adapter_contracts import (
    VectorAdapterDelta,
    VectorAdapterState,
)


def _build_base_state() -> VectorAdapterState:
    return VectorAdapterState(
        schema_version="vector_adapter_state.v1",
        adapter_kind="diagonal_scale",
        model_id="tracemind-embed",
        model_revision="rev_000",
        training_scope="adapter_only",
        dimension_scales=[1.0, 1.0],
        updated_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def _build_update(
    *,
    deltas: list[float],
    example_count: int = 1,
) -> VectorAdapterDelta:
    return VectorAdapterDelta(
        schema_version="vector_adapter_delta.v1",
        adapter_kind="diagonal_scale",
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        dimension_deltas=deltas,
        example_count=example_count,
        mean_confidence=0.9,
        mean_margin=0.2,
        label_counts={"anxiety": example_count},
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def test_diagonal_scale_aggregation_uses_shared_default_config() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="diagonal_scale",
        backend_name="fedavg",
    )

    assert isinstance(backend, DiagonalScaleAggregationService)
    assert backend.config == DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG


def test_diagonal_scale_aggregation_applies_backend_overrides() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="diagonal_scale",
        backend_name="fedavg",
        overrides={"min_scale": 0.9, "max_scale": 1.1},
    )

    result = backend.aggregate(
        base_state=_build_base_state(),
        update_payloads=(
            _build_update(deltas=[0.5, -0.5], example_count=1),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    assert isinstance(backend, DiagonalScaleAggregationService)
    assert backend.config.min_scale == 0.9
    assert backend.config.max_scale == 1.1
    assert result.next_state.dimension_scales == [1.1, 0.9]
