"""Shared adapter aggregation backend tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from main_server.src.services.federation.rounds.aggregation.executor import (
    MethodAggregationBackend,
)
from main_server.src.services.federation.rounds.aggregation.registry import (
    build_shared_adapter_aggregation_backend,
    list_shared_adapter_aggregation_backend_catalog_entries,
)
from methods.adaptation.diagonal_scale.aggregation.fedavg import (
    DEFAULT_DIAGONAL_SCALE_MAX_SCALE,
    DEFAULT_DIAGONAL_SCALE_MIN_SCALE,
)
from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
)
from methods.adaptation.lora_classifier.update.partitioned_tensor_artifact import (
    build_partitioned_delta_tensor_artifact,
)
from methods.federated.aggregation.fedavg.strategy import (
    FedAvgAggregationStrategy,
)
from methods.federated.aggregation.registry import build_federated_aggregation_strategy
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadDelta,
    ClassifierHeadState,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    VectorAdapterDelta,
    VectorAdapterState,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
    LoraClassifierState,
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

    assert (
        entries["diagonal_scale.fedavg"].implementation_module
        == "methods.adaptation.diagonal_scale.aggregation.fedavg"
    )
    assert (
        entries["classifier_head.fedavg"].implementation_module
        == "methods.adaptation.classifier_head.aggregation.fedavg"
    )
    assert (
        entries["lora_classifier.fedavg"].implementation_module
        == "methods.adaptation.lora_classifier.aggregation.fedavg"
    )
    assert (
        entries["lora_classifier.fedavg"].metadata[
            "requires_inline_or_materialized_artifacts"
        ]
        is True
    )
    assert (
        entries["lora_classifier.partitioned_delta_average"].implementation_module
        == "methods.adaptation.lora_classifier.aggregation.partitioned_delta_average"
    )
    assert (
        entries["lora_classifier.partitioned_delta_average"].metadata[
            "requires_partitioned_deltas"
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


def test_lora_classifier_fedavg_aggregation_publishes_next_state_refs(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/lora_adapter",
        {
            "lora_parameters": {
                "encoder.q_proj.lora_A": [1.0, 1.0],
                "encoder.extra.lora_A": [0.5],
            }
        },
    )
    artifact_store.save_json_artifact(
        "rev_000/classifier_head",
        {
            "classifier_head_weights": {
                "anxiety": [1.0, 0.0],
                "normal": [0.0, 1.0],
            },
            "classifier_head_biases": {
                "anxiety": 0.1,
                "normal": -0.1,
            },
        },
    )
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="fedavg",
        overrides={"artifact_ref_prefix": "server-aggregate://test_lora"},
        artifact_store=artifact_store,
    )
    base_state = _build_lora_state(
        lora_adapter_artifact_ref=artifact_store.ref_for_artifact(
            "rev_000/lora_adapter"
        ),
        classifier_head_artifact_ref=artifact_store.ref_for_artifact(
            "rev_000/classifier_head"
        ),
    )

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
    assert result.aggregated_artifacts
    lora_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.lora_adapter_artifact_ref
    )
    head_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.classifier_head_artifact_ref
    )
    assert lora_artifact["lora_parameters"] == {
        "encoder.extra.lora_A": pytest.approx([0.5]),
        "encoder.q_proj.lora_A": pytest.approx([1.1333333333333333, 1.3]),
        "encoder.q_proj.lora_B": pytest.approx([0.13333333333333333, 0.0]),
    }
    assert lora_artifact["applied_lora_parameter_deltas"][
        "encoder.q_proj.lora_A"
    ] == pytest.approx([0.13333333333333333, 0.3])
    assert head_artifact["classifier_head_weights"]["anxiety"] == pytest.approx(
        [1.1666666666666667, 0.0]
    )
    assert head_artifact["classifier_head_weights"]["normal"] == pytest.approx(
        [-0.16666666666666669, 1.0]
    )
    assert head_artifact["classifier_head_biases"]["anxiety"] == pytest.approx(0.14)
    assert head_artifact["classifier_head_biases"]["normal"] == pytest.approx(
        -0.13333333333333333
    )
    assert result.aggregated_metrics["client_count"] == 2.0
    assert result.aggregated_metrics["example_count"] == 3.0
    assert result.aggregated_metrics["mean_confidence"] == pytest.approx(0.8)
    assert result.aggregated_metrics["lora_parameter_count"] == 2.0
    assert result.update_count == 2


def test_lora_classifier_partitioned_delta_average_publishes_next_state_refs(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/lora_adapter",
        {"lora_parameters": {"encoder.q_proj.lora_A": [1.0, 1.0]}},
    )
    artifact_store.save_json_artifact(
        "rev_000/classifier_head",
        {
            "classifier_head_weights": {
                "anxiety": [1.0, 0.0],
                "normal": [0.0, 1.0],
            },
            "classifier_head_biases": {"anxiety": 0.1, "normal": -0.1},
        },
    )
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="partitioned_delta_average",
        overrides={"artifact_ref_prefix": "server-aggregate://partitioned_lora"},
        artifact_store=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_lora_state(
            lora_adapter_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/lora_adapter"
            ),
            classifier_head_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/classifier_head"
            ),
        ),
        update_payloads=(
            make_lora_classifier_delta_payload(
                model_id="tracemind-lora",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_lora_backbone(),
                lora_config=_lora_config(),
                label_schema=("anxiety", "normal"),
                example_count=2,
                partitioned_deltas={
                    "sigma": {
                        "lora_parameter_deltas": {"encoder.q_proj.lora_A": [0.2, 0.0]},
                        "classifier_head_weight_deltas": {
                            "anxiety": [0.1, 0.0],
                            "normal": [-0.1, 0.0],
                        },
                    },
                    "psi": {
                        "lora_parameter_deltas": {"encoder.q_proj.lora_A": [0.1, 0.3]},
                        "classifier_head_weight_deltas": {
                            "anxiety": [0.2, 0.2],
                            "normal": [-0.2, -0.2],
                        },
                        "classifier_head_bias_deltas": {"anxiety": 0.03},
                    },
                },
                delta_format="partitioned_update",
                mean_confidence=0.9,
                mean_margin=0.2,
                created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    assert isinstance(backend, MethodAggregationBackend)
    assert backend.adapter_kind == "lora_classifier"
    lora_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.lora_adapter_artifact_ref or ""
    )
    head_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.classifier_head_artifact_ref or ""
    )
    assert lora_artifact["lora_parameters"]["encoder.q_proj.lora_A"] == pytest.approx(
        [1.3, 1.3]
    )
    assert head_artifact["classifier_head_weights"]["anxiety"] == pytest.approx(
        [1.3, 0.2]
    )
    assert head_artifact["classifier_head_weights"]["normal"] == pytest.approx(
        [-0.3, 0.8]
    )
    assert head_artifact["classifier_head_biases"] == pytest.approx(
        {"anxiety": 0.13, "normal": -0.1}
    )
    assert result.aggregated_metrics["server_update_partitioned"] == 1.0
    assert result.aggregated_metrics["partition_count_total"] == 2.0
    assert result.update_count == 1


def test_lora_classifier_partitioned_delta_average_reads_partitioned_artifact_ref(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/lora_adapter",
        {"lora_parameters": {"encoder.q_proj.lora_A": [1.0, 1.0]}},
    )
    artifact_store.save_json_artifact(
        "rev_000/classifier_head",
        {
            "classifier_head_weights": {
                "anxiety": [1.0, 0.0],
                "normal": [0.0, 1.0],
            },
            "classifier_head_biases": {"anxiety": 0.1, "normal": -0.1},
        },
    )
    partitioned_ref = artifact_store.ref_for_artifact(
        "client_updates/round_0001/agent_01/update_001/partitioned_delta"
    )
    artifact_store.save_json_artifact_ref(
        artifact_ref=partitioned_ref,
        payload={
            "partitions": {
                "sigma": {
                    "lora_parameter_deltas": {"encoder.q_proj.lora_A": [0.2, 0.0]},
                    "classifier_head_weight_deltas": {
                        "anxiety": [0.1, 0.0],
                        "normal": [-0.1, 0.0],
                    },
                },
                "psi": {
                    "lora_parameter_deltas": {"encoder.q_proj.lora_A": [0.1, 0.3]},
                    "classifier_head_weight_deltas": {
                        "anxiety": [0.2, 0.2],
                        "normal": [-0.2, -0.2],
                    },
                    "classifier_head_bias_deltas": {"anxiety": 0.03},
                },
            }
        },
    )
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="partitioned_delta_average",
        overrides={"artifact_ref_prefix": "server-aggregate://partitioned_lora"},
        artifact_store=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_lora_state(
            lora_adapter_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/lora_adapter"
            ),
            classifier_head_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/classifier_head"
            ),
        ),
        update_payloads=(
            make_lora_classifier_delta_payload(
                model_id="tracemind-lora",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_lora_backbone(),
                lora_config=_lora_config(),
                label_schema=("anxiety", "normal"),
                example_count=2,
                partitioned_deltas_artifact_ref=partitioned_ref,
                delta_format="server_uploaded_artifact_ref",
                delta_l2_norm=1.0,
                mean_confidence=0.9,
                mean_margin=0.2,
                created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    lora_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.lora_adapter_artifact_ref or ""
    )
    assert lora_artifact["lora_parameters"]["encoder.q_proj.lora_A"] == pytest.approx(
        [1.3, 1.3]
    )
    assert result.aggregated_metrics["server_update_partitioned"] == 1.0


def test_lora_classifier_partitioned_delta_average_reads_tensor_artifact_ref(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/lora_adapter",
        {"lora_parameters": {"encoder.q_proj.lora_A": [1.0, 1.0]}},
    )
    artifact_store.save_json_artifact(
        "rev_000/classifier_head",
        {
            "classifier_head_weights": {
                "anxiety": [1.0, 0.0],
                "normal": [0.0, 1.0],
            },
            "classifier_head_biases": {"anxiety": 0.1, "normal": -0.1},
        },
    )
    partitioned_ref = artifact_store.ref_for_artifact(
        "client_updates/round_0001/agent_01/update_001/partitioned_delta"
    )
    tensors, metadata = build_partitioned_delta_tensor_artifact(
        {
            "sigma": LoraClassifierPartitionDelta(
                partition_name="sigma",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.2, 0.0]},
                classifier_head_weight_deltas={
                    "anxiety": [0.1, 0.0],
                    "normal": [-0.1, 0.0],
                },
                classifier_head_bias_deltas={},
            ),
            "psi": LoraClassifierPartitionDelta(
                partition_name="psi",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, 0.3]},
                classifier_head_weight_deltas={
                    "anxiety": [0.2, 0.2],
                    "normal": [-0.2, -0.2],
                },
                classifier_head_bias_deltas={"anxiety": 0.03},
            ),
        }
    )
    artifact_store.save_safetensors_artifact_ref(
        artifact_ref=partitioned_ref,
        tensors=tensors,
        metadata=metadata,
    )
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="partitioned_delta_average",
        overrides={"artifact_ref_prefix": "server-aggregate://partitioned_lora"},
        artifact_store=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_lora_state(
            lora_adapter_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/lora_adapter"
            ),
            classifier_head_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/classifier_head"
            ),
        ),
        update_payloads=(
            make_lora_classifier_delta_payload(
                model_id="tracemind-lora",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_lora_backbone(),
                lora_config=_lora_config(),
                label_schema=("anxiety", "normal"),
                example_count=2,
                partitioned_deltas_artifact_ref=partitioned_ref,
                delta_format="server_uploaded_artifact_ref",
                delta_l2_norm=1.0,
                mean_confidence=0.9,
                mean_margin=0.2,
                created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    lora_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.lora_adapter_artifact_ref or ""
    )
    assert lora_artifact["lora_parameters"]["encoder.q_proj.lora_A"] == pytest.approx(
        [1.3, 1.3]
    )
    assert result.aggregated_metrics["server_update_partitioned"] == 1.0


def test_lora_classifier_fedavg_two_rounds_accumulates_global_snapshot(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="fedavg",
        overrides={"artifact_ref_prefix": "server-aggregate://test_lora"},
        artifact_store=artifact_store,
    )

    first_result = backend.aggregate(
        base_state=_build_lora_state(),
        update_payloads=(
            _build_lora_update(
                lora_deltas={"encoder.q_proj.lora_A": [0.2, 0.4]},
                head_weight_deltas={
                    "anxiety": [0.2, -0.1],
                    "normal": [-0.2, 0.1],
                },
                head_bias_deltas={"anxiety": 0.05, "normal": -0.05},
                example_count=2,
                mean_confidence=0.9,
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )
    second_result = backend.aggregate(
        base_state=first_result.next_state,
        update_payloads=(
            _build_lora_update(
                base_model_revision="rev_001",
                lora_deltas={
                    "encoder.q_proj.lora_A": [0.3, -0.3],
                    "encoder.q_proj.lora_B": [0.1, 0.1],
                },
                head_weight_deltas={
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                head_bias_deltas={"anxiety": 0.02},
                example_count=1,
                mean_confidence=0.7,
            ),
            _build_lora_update(
                base_model_revision="rev_001",
                lora_deltas={
                    "encoder.q_proj.lora_A": [0.0, 0.3],
                    "encoder.q_proj.lora_B": [0.2, -0.2],
                },
                head_weight_deltas={
                    "anxiety": [0.4, -0.1],
                    "normal": [-0.4, 0.1],
                },
                head_bias_deltas={"normal": -0.03},
                example_count=3,
                mean_confidence=0.8,
            ),
        ),
        next_model_revision="rev_002",
        aggregated_at=datetime(2026, 4, 8, 2, tzinfo=timezone.utc),
    )

    first_lora = artifact_store.load_json_artifact(
        artifact_ref=first_result.next_state.lora_adapter_artifact_ref
    )
    first_head = artifact_store.load_json_artifact(
        artifact_ref=first_result.next_state.classifier_head_artifact_ref
    )
    second_lora = artifact_store.load_json_artifact(
        artifact_ref=second_result.next_state.lora_adapter_artifact_ref
    )
    second_head = artifact_store.load_json_artifact(
        artifact_ref=second_result.next_state.classifier_head_artifact_ref
    )

    _assert_vector_mapping_accumulates(
        before=first_lora["lora_parameters"],
        delta=second_lora["applied_lora_parameter_deltas"],
        after=second_lora["lora_parameters"],
    )
    _assert_vector_mapping_accumulates(
        before=first_head["classifier_head_weights"],
        delta=second_head["applied_classifier_head_weight_deltas"],
        after=second_head["classifier_head_weights"],
    )
    _assert_scalar_mapping_accumulates(
        before=first_head["classifier_head_biases"],
        delta=second_head["applied_classifier_head_bias_deltas"],
        after=second_head["classifier_head_biases"],
    )
    assert second_result.next_state.model_revision == "rev_002"


def test_lora_classifier_fedavg_materializes_server_owned_artifact_updates(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "u1/lora_delta",
        {
            "lora_parameter_deltas": {
                "encoder.q_proj.lora_A": [0.2, 0.4],
                "encoder.q_proj.lora_B": [0.1, -0.1],
            }
        },
    )
    artifact_store.save_json_artifact(
        "u1/classifier_head_delta",
        {
            "classifier_head_weight_deltas": {
                "anxiety": [0.2, -0.1],
                "normal": [-0.2, 0.1],
            },
            "classifier_head_bias_deltas": {
                "anxiety": 0.05,
                "normal": -0.05,
            },
        },
    )
    backend = MethodAggregationBackend(
        strategy=build_federated_aggregation_strategy(
            adapter_kind="lora_classifier",
            method_name="fedavg",
            overrides={"artifact_ref_prefix": "server-aggregate://test_lora"},
        ),
        overrides={"artifact_ref_prefix": "server-aggregate://test_lora"},
        artifact_loader=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_lora_state(),
        update_payloads=(
            make_lora_classifier_delta_payload(
                model_id="tracemind-lora",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_lora_backbone(),
                lora_config=_lora_config(),
                label_schema=("anxiety", "normal"),
                example_count=2,
                lora_delta_artifact_ref=artifact_store.ref_for_artifact(
                    "u1/lora_delta"
                ),
                classifier_head_delta_artifact_ref=(
                    artifact_store.ref_for_artifact("u1/classifier_head_delta")
                ),
                delta_format="server_uploaded_artifact_ref",
                mean_confidence=0.9,
                mean_margin=0.2,
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    assert result.next_state.lora_adapter_artifact_ref == (
        "server-aggregate://test_lora/rev_001/lora_adapter"
    )
    assert result.aggregated_metrics["client_count"] == 1.0
    assert result.aggregated_metrics["lora_parameter_count"] == 2.0
    assert result.aggregated_metrics["classifier_head_label_count"] == 2.0


def test_aggregation_artifact_store_defaults_to_main_server_state_root() -> None:
    store = AggregationArtifactStore()

    assert store.state_root == (
        Path(__file__).resolve().parents[2] / "state" / "aggregation_artifacts"
    )
    with pytest.raises(ValueError, match="path traversal"):
        store.ref_for_artifact("../escape")


def test_lora_classifier_fedavg_rejects_agent_local_artifact_only_updates() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="lora_classifier",
        backend_name="fedavg",
    )

    with pytest.raises(FileNotFoundError, match="Unsupported aggregation artifact ref"):
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


def _build_lora_state(
    *,
    lora_adapter_artifact_ref: str | None = None,
    classifier_head_artifact_ref: str | None = None,
) -> LoraClassifierState:
    return make_lora_classifier_state_payload(
        model_id="tracemind-lora",
        model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_lora_backbone(),
        lora_config=_lora_config(),
        label_schema=("anxiety", "normal"),
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        updated_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def _build_lora_update(
    *,
    base_model_revision: str = "rev_000",
    lora_deltas: dict[str, list[float]],
    head_weight_deltas: dict[str, list[float]],
    head_bias_deltas: dict[str, float],
    example_count: int,
    mean_confidence: float,
) -> LoraClassifierDelta:
    return make_lora_classifier_delta_payload(
        model_id="tracemind-lora",
        base_model_revision=base_model_revision,
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


def _assert_vector_mapping_accumulates(
    *,
    before: object,
    delta: object,
    after: object,
) -> None:
    assert isinstance(before, Mapping)
    assert isinstance(delta, Mapping)
    assert isinstance(after, Mapping)
    for key in sorted(set(before) | set(delta)):
        before_values = before.get(key, [])
        delta_values = delta.get(key, [])
        after_values = after[key]
        assert isinstance(before_values, Sequence)
        assert isinstance(delta_values, Sequence)
        assert isinstance(after_values, Sequence)
        if not before_values:
            assert list(after_values) == pytest.approx(list(delta_values))
            continue
        if not delta_values:
            assert list(after_values) == pytest.approx(list(before_values))
            continue
        assert list(after_values) == pytest.approx(
            [
                float(before_value) + float(delta_value)
                for before_value, delta_value in zip(
                    before_values,
                    delta_values,
                    strict=True,
                )
            ]
        )


def _assert_scalar_mapping_accumulates(
    *,
    before: object,
    delta: object,
    after: object,
) -> None:
    assert isinstance(before, Mapping)
    assert isinstance(delta, Mapping)
    assert isinstance(after, Mapping)
    for key in sorted(set(before) | set(delta)):
        assert float(after[key]) == pytest.approx(
            float(before.get(key, 0.0)) + float(delta.get(key, 0.0))
        )
