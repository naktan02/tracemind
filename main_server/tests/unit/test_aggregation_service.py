"""Shared adapter aggregation backend tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from main_server.src.services.federation.rounds.aggregation.executor import (
    MethodAggregationBackend,
)
from main_server.src.services.federation.rounds.aggregation.registry import (
    build_shared_adapter_aggregation_backend,
    list_shared_adapter_aggregation_backend_catalog_entries,
)
from methods.adaptation.diagonal_scale.fedavg_projection import (
    DEFAULT_DIAGONAL_SCALE_MAX_SCALE,
    DEFAULT_DIAGONAL_SCALE_MIN_SCALE,
)
from methods.federated.aggregation.fedavg.strategy import (
    FedAvgAggregationStrategy,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    ClassifierHeadState,
    LoraClassifierDelta,
    LoraClassifierState,
    VectorAdapterDelta,
    VectorAdapterState,
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
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

    assert isinstance(backend, MethodAggregationBackend)
    assert isinstance(backend.strategy, FedAvgAggregationStrategy)
    assert backend.adapter_kind == "diagonal_scale"
    assert backend.strategy.overrides is None
    assert DEFAULT_DIAGONAL_SCALE_MIN_SCALE == 0.75
    assert DEFAULT_DIAGONAL_SCALE_MAX_SCALE == 1.25


def test_aggregation_backend_catalog_points_to_methods_core() -> None:
    entries = {
        entry.item_name: entry
        for entry in list_shared_adapter_aggregation_backend_catalog_entries()
    }

    assert entries["diagonal_scale.fedavg"].implementation_module.endswith(
        "diagonal_scale_fedavg"
    )
    assert entries["classifier_head.fedavg"].implementation_module.endswith(
        "classifier_head_fedavg"
    )
    assert entries["lora_classifier.fedavg"].implementation_module.endswith(
        "lora_classifier_fedavg"
    )
    assert (
        entries["lora_classifier.fedavg"].metadata[
            "requires_inline_or_materialized_artifacts"
        ]
        is True
    )


def test_diagonal_scale_aggregation_applies_backend_overrides() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="diagonal_scale",
        backend_name="fedavg",
        overrides={"min_scale": 0.9, "max_scale": 1.1},
    )

    result = backend.aggregate(
        base_state=_build_base_state(),
        update_payloads=(_build_update(deltas=[0.5, -0.5], example_count=1),),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    assert isinstance(backend, MethodAggregationBackend)
    assert backend.adapter_kind == "diagonal_scale"
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

    assert isinstance(backend, MethodAggregationBackend)
    assert backend.adapter_kind == "classifier_head"
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


def test_lora_classifier_fedavg_aggregation_publishes_next_state_refs() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="fedavg",
        overrides={"artifact_ref_prefix": "server-aggregate://test_lora"},
    )
    base_state = _build_lora_state()

    result = backend.aggregate(
        base_state=base_state,
        update_payloads=(
            _build_lora_update(
                lora_deltas={
                    "encoder.q_proj.lora_A": [0.2, 0.4],
                    "encoder.q_proj.lora_B": [0.1, -0.1],
                },
                head_weight_deltas={
                    "anxiety": [0.2, -0.1],
                    "normal": [-0.2, 0.1],
                },
                head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
                example_count=2,
                mean_confidence=0.9,
            ),
            _build_lora_update(
                lora_deltas={
                    "encoder.q_proj.lora_A": [0.0, 0.1],
                    "encoder.q_proj.lora_B": [0.2, 0.2],
                },
                head_weight_deltas={
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                head_bias_deltas={"anxiety": 0.02},
                example_count=1,
                mean_confidence=0.6,
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    assert isinstance(backend, MethodAggregationBackend)
    assert backend.adapter_kind == "lora_classifier"
    assert isinstance(result.next_state, LoraClassifierState)
    assert result.next_state.model_revision == "rev_001"
    assert result.next_state.lora_adapter_artifact_ref == (
        "server-aggregate://test_lora/rev_001/lora_adapter"
    )
    assert result.next_state.classifier_head_artifact_ref == (
        "server-aggregate://test_lora/rev_001/classifier_head"
    )
    assert result.next_state.artifact_format == "server_aggregated_artifact_ref"
    assert result.aggregated_metrics["client_count"] == 2.0
    assert result.aggregated_metrics["example_count"] == 3.0
    assert result.aggregated_metrics["mean_confidence"] == pytest.approx(0.8)
    assert result.aggregated_metrics["lora_parameter_count"] == 2.0
    assert result.update_count == 2


def test_lora_classifier_fedavg_rejects_artifact_only_updates() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="fedavg",
    )

    with pytest.raises(ValueError, match="Artifact-ref-only updates"):
        backend.aggregate(
            base_state=_build_lora_state(),
            update_payloads=(
                make_lora_classifier_delta_payload(
                    model_id="tracemind-lora",
                    base_model_revision="rev_000",
                    training_scope="adapter_only",
                    backbone=_lora_backbone(),
                    lora_config=_lora_config(),
                    label_schema=("anxiety", "normal"),
                    example_count=1,
                    lora_delta_artifact_ref="agent-local://u1/lora_delta",
                    classifier_head_delta_artifact_ref=(
                        "agent-local://u1/classifier_head_delta"
                    ),
                    delta_format="agent_local_artifact_ref",
                    mean_confidence=0.9,
                ),
            ),
            next_model_revision="rev_001",
            aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
        )


def _lora_backbone() -> dict[str, str | int]:
    return {
        "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "backbone_revision": "main",
        "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "tokenizer_revision": "main",
        "pooling": "mean",
        "max_length": 256,
        "task_prefix": "",
    }


def _lora_config() -> dict[str, str | int | float | bool]:
    return {
        "peft_adapter_name": "lora",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "bias": "none",
        "target_modules": "all-linear",
        "use_rslora": False,
    }


def _build_lora_state() -> LoraClassifierState:
    return make_lora_classifier_state_payload(
        model_id="tracemind-lora",
        model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_lora_backbone(),
        lora_config=_lora_config(),
        label_schema=("anxiety", "normal"),
        lora_adapter_artifact_ref="shared://rev_000/lora_adapter",
        classifier_head_artifact_ref="shared://rev_000/classifier_head",
        updated_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def _build_lora_update(
    *,
    lora_deltas: dict[str, list[float]],
    head_weight_deltas: dict[str, list[float]],
    head_bias_deltas: dict[str, float],
    example_count: int,
    mean_confidence: float,
) -> LoraClassifierDelta:
    return make_lora_classifier_delta_payload(
        model_id="tracemind-lora",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_lora_backbone(),
        lora_config=_lora_config(),
        label_schema=("anxiety", "normal"),
        example_count=example_count,
        lora_parameter_deltas=lora_deltas,
        classifier_head_weight_deltas=head_weight_deltas,
        classifier_head_bias_deltas=head_bias_deltas,
        delta_format="inline_delta",
        mean_confidence=mean_confidence,
        mean_margin=0.2,
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )
