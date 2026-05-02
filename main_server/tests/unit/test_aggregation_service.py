"""Shared adapter aggregation backend tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from main_server.src.services.federation.rounds import (
    DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG,
    ClassifierHeadFedAvgAggregationService,
    DiagonalScaleAggregationService,
    build_shared_adapter_aggregation_backend,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    ClassifierHeadState,
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


def test_classifier_head_fedavg_aggregation_updates_weights_and_biases() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="classifier_head",
        backend_name="fedavg",
    )

    result = backend.aggregate(
        base_state=ClassifierHeadState(
            schema_version="classifier_head_state.v1",
            adapter_kind="classifier_head",
            model_id="tracemind-embed",
            model_revision="rev_000",
            training_scope="head_only",
            updated_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
            label_weights={
                "anxiety": [1.0, 0.0],
                "normal": [0.0, 1.0],
            },
            label_biases={"anxiety": 0.1, "normal": -0.1},
        ),
        update_payloads=(
            ClassifierHeadDelta(
                schema_version="classifier_head_delta.v1",
                adapter_kind="classifier_head",
                model_id="tracemind-embed",
                base_model_revision="rev_000",
                training_scope="head_only",
                example_count=2,
                created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                label_weight_deltas={
                    "anxiety": [0.2, -0.1],
                    "normal": [-0.2, 0.1],
                },
                label_bias_deltas={"anxiety": 0.05, "normal": -0.05},
                mean_confidence=0.9,
                mean_margin=0.3,
                label_counts={"anxiety": 2},
            ),
            ClassifierHeadDelta(
                schema_version="classifier_head_delta.v1",
                adapter_kind="classifier_head",
                model_id="tracemind-embed",
                base_model_revision="rev_000",
                training_scope="head_only",
                example_count=1,
                created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                label_weight_deltas={
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                label_bias_deltas={"anxiety": 0.03, "normal": -0.03},
                mean_confidence=0.8,
                mean_margin=0.2,
                label_counts={"normal": 1},
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    assert isinstance(backend, ClassifierHeadFedAvgAggregationService)
    assert result.next_state.label_weights["anxiety"] == pytest.approx(
        [1.1666666666666667, 0.0]
    )
    assert result.next_state.label_weights["normal"] == pytest.approx(
        [-0.16666666666666669, 1.0]
    )
    assert result.next_state.label_biases["anxiety"] == pytest.approx(
        0.14333333333333334
    )
    assert result.next_state.label_biases["normal"] == pytest.approx(
        -0.14333333333333334
    )
