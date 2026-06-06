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
from methods.adaptation.peft_text_encoder.update import (
    merged_tensor_artifact as merged_artifacts,
)
from methods.adaptation.peft_text_encoder.update import (
    partitioned_tensor_artifact as partitioned_artifacts,
)
from methods.adaptation.peft_text_encoder.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
)
from methods.federated.aggregation.registry import build_federated_aggregation_strategy
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadDelta,
    ClassifierHeadState,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierDelta,
    PeftClassifierState,
)


def test_aggregation_backend_catalog_points_to_methods_core() -> None:
    entries = {
        entry.item_name: entry
        for entry in list_shared_adapter_aggregation_backend_catalog_entries()
    }

    assert "diagonal_scale.fedavg" not in entries
    assert (
        entries["classifier_head.fedavg"].implementation_module
        == "methods.classification.linear_head.aggregation."
        "linear_head_fedavg_projection"
    )
    assert (
        entries["peft_classifier.fedavg"].implementation_module
        == "methods.adaptation.peft_text_encoder.aggregation."
        "peft_encoder_fedavg_projection"
    )
    assert (
        entries["peft_classifier.fedavg"].metadata[
            "requires_inline_or_materialized_artifacts"
        ]
        is True
    )
    assert (
        entries["peft_classifier.partitioned_delta_average"].implementation_module
        == "methods.adaptation.peft_text_encoder.aggregation."
        "peft_encoder_partitioned_projection"
    )
    assert (
        entries["peft_classifier.partitioned_delta_average"].metadata[
            "requires_partitioned_deltas"
        ]
        is True
    )


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
            ),
        ),
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
    )

    assert isinstance(backend, MethodAggregationBackend)
    assert backend.adapter_kind == "classifier_head"
    assert result.next_state.label_weights["anxiety"] == pytest.approx([1.15, 0.05])
    assert result.next_state.label_weights["normal"] == pytest.approx([-0.15, 0.95])
    assert result.next_state.label_biases["anxiety"] == pytest.approx(0.14)
    assert result.next_state.label_biases["normal"] == pytest.approx(-0.14)


def test_peft_classifier_fedavg_aggregation_publishes_next_state_refs(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/peft_adapter",
        {
            "peft_parameters": {
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
        adapter_kind="peft_classifier",
        backend_name="fedavg",
        overrides={"artifact_ref_prefix": "server-aggregate://test_peft"},
        artifact_store=artifact_store,
    )
    base_state = _build_peft_state(
        peft_adapter_artifact_ref=artifact_store.ref_for_artifact(
            "rev_000/peft_adapter"
        ),
        classifier_head_artifact_ref=artifact_store.ref_for_artifact(
            "rev_000/classifier_head"
        ),
    )

    result = backend.aggregate(
        base_state=base_state,
        update_payloads=(
            _build_peft_update(
                peft_deltas={
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
            _build_peft_update(
                peft_deltas={
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
    assert backend.adapter_kind == "peft_classifier"
    assert isinstance(result.next_state, PeftClassifierState)
    assert result.next_state.model_revision == "rev_001"
    assert result.next_state.peft_adapter_artifact_ref == (
        "server-aggregate://test_peft/rev_001/peft_adapter"
    )
    assert result.next_state.classifier_head_artifact_ref == (
        "server-aggregate://test_peft/rev_001/classifier_head"
    )
    assert result.next_state.artifact_format == "server_aggregated_artifact_ref"
    assert result.aggregated_artifacts
    peft_tensors, peft_metadata = artifact_store.load_safetensors_artifact(
        artifact_ref=result.next_state.peft_adapter_artifact_ref
    )
    peft_artifact = merged_artifacts.parse_peft_adapter_state_tensor_artifact(
        tensors=peft_tensors,
        metadata=peft_metadata,
    )
    applied_peft_deltas = (
        merged_artifacts.parse_applied_peft_parameter_deltas_tensor_artifact(
            tensors=peft_tensors,
            metadata=peft_metadata,
        )
    )
    head_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.classifier_head_artifact_ref
    )
    assert (
        result.aggregated_artifacts[result.next_state.peft_adapter_artifact_ref][
            "artifact_format"
        ]
        == "safetensors"
    )
    assert peft_artifact == {
        "encoder.extra.lora_A": pytest.approx([0.5]),
        "encoder.q_proj.lora_A": pytest.approx([1.1, 1.25]),
        "encoder.q_proj.lora_B": pytest.approx([0.15, 0.05]),
    }
    assert applied_peft_deltas["encoder.q_proj.lora_A"] == pytest.approx([0.1, 0.25])
    assert head_artifact["classifier_head_weights"]["anxiety"] == pytest.approx(
        [1.15, 0.05]
    )
    assert head_artifact["classifier_head_weights"]["normal"] == pytest.approx(
        [-0.15, 0.95]
    )
    assert head_artifact["classifier_head_biases"]["anxiety"] == pytest.approx(0.135)
    assert head_artifact["classifier_head_biases"]["normal"] == pytest.approx(-0.125)
    assert result.aggregated_metrics["client_count"] == 2.0
    assert result.aggregated_metrics["example_count"] == 3.0
    assert result.aggregated_metrics["mean_confidence"] == pytest.approx(0.8)
    assert result.aggregated_metrics["aggregation_weight_policy_example_count"] == 0.0
    assert result.aggregated_metrics["peft_parameter_count"] == 2.0
    assert result.update_count == 2


def test_peft_classifier_partitioned_delta_average_publishes_next_state_refs(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/peft_adapter",
        {"peft_parameters": {"encoder.q_proj.lora_A": [1.0, 1.0]}},
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
        adapter_kind="peft_classifier",
        backend_name="partitioned_delta_average",
        overrides={"artifact_ref_prefix": "server-aggregate://partitioned_peft"},
        artifact_store=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_peft_state(
            peft_adapter_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/peft_adapter"
            ),
            classifier_head_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/classifier_head"
            ),
        ),
        update_payloads=(
            make_peft_classifier_delta_payload(
                model_id="tracemind-peft",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_peft_backbone(),
                peft_adapter_config=_peft_adapter_config(),
                label_schema=("anxiety", "normal"),
                example_count=2,
                partitioned_deltas={
                    "sigma": {
                        "peft_parameter_deltas": {"encoder.q_proj.lora_A": [0.2, 0.0]},
                        "classifier_head_weight_deltas": {
                            "anxiety": [0.1, 0.0],
                            "normal": [-0.1, 0.0],
                        },
                    },
                    "psi": {
                        "peft_parameter_deltas": {"encoder.q_proj.lora_A": [0.1, 0.3]},
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
    assert backend.adapter_kind == "peft_classifier"
    peft_artifact = _load_peft_state_artifact(
        artifact_store, artifact_ref=result.next_state.peft_adapter_artifact_ref or ""
    )
    partitioned_peft_artifact = _load_partitioned_peft_state_artifact(
        artifact_store,
        artifact_ref=result.next_state.peft_adapter_artifact_ref or "",
    )
    head_artifact = artifact_store.load_json_artifact(
        artifact_ref=result.next_state.classifier_head_artifact_ref or ""
    )
    assert peft_artifact["encoder.q_proj.lora_A"] == pytest.approx([1.3, 1.3])
    assert partitioned_peft_artifact["sigma"]["encoder.q_proj.lora_A"] == pytest.approx(
        [1.2, 1.0]
    )
    assert partitioned_peft_artifact["psi"]["encoder.q_proj.lora_A"] == pytest.approx(
        [1.1, 1.3]
    )
    assert head_artifact["classifier_head_weights"]["anxiety"] == pytest.approx(
        [1.3, 0.2]
    )
    assert head_artifact["partitioned_classifier_head_weights"]["sigma"][
        "anxiety"
    ] == pytest.approx([1.1, 0.0])
    assert head_artifact["partitioned_classifier_head_weights"]["psi"][
        "anxiety"
    ] == pytest.approx([1.2, 0.2])
    assert head_artifact["classifier_head_weights"]["normal"] == pytest.approx(
        [-0.3, 0.8]
    )
    assert head_artifact["classifier_head_biases"] == pytest.approx(
        {"anxiety": 0.13, "normal": -0.1}
    )
    assert result.aggregated_metrics["server_update_partitioned"] == 1.0
    assert result.aggregated_metrics["partition_count_total"] == 2.0
    assert result.aggregated_metrics["partitioned_global_state_count"] == 2.0
    assert result.update_count == 1


def test_peft_classifier_partitioned_delta_average_reads_partitioned_artifact_ref(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/peft_adapter",
        {"peft_parameters": {"encoder.q_proj.lora_A": [1.0, 1.0]}},
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
                    "peft_parameter_deltas": {"encoder.q_proj.lora_A": [0.2, 0.0]},
                    "classifier_head_weight_deltas": {
                        "anxiety": [0.1, 0.0],
                        "normal": [-0.1, 0.0],
                    },
                },
                "psi": {
                    "peft_parameter_deltas": {"encoder.q_proj.lora_A": [0.1, 0.3]},
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
        adapter_kind="peft_classifier",
        backend_name="partitioned_delta_average",
        overrides={"artifact_ref_prefix": "server-aggregate://partitioned_peft"},
        artifact_store=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_peft_state(
            peft_adapter_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/peft_adapter"
            ),
            classifier_head_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/classifier_head"
            ),
        ),
        update_payloads=(
            make_peft_classifier_delta_payload(
                model_id="tracemind-peft",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_peft_backbone(),
                peft_adapter_config=_peft_adapter_config(),
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

    peft_artifact = _load_peft_state_artifact(
        artifact_store, artifact_ref=result.next_state.peft_adapter_artifact_ref or ""
    )
    assert peft_artifact["encoder.q_proj.lora_A"] == pytest.approx([1.3, 1.3])
    assert result.aggregated_metrics["server_update_partitioned"] == 1.0


def test_peft_classifier_partitioned_delta_average_reads_tensor_artifact_ref(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "rev_000/peft_adapter",
        {"peft_parameters": {"encoder.q_proj.lora_A": [1.0, 1.0]}},
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
    tensors, metadata = partitioned_artifacts.build_partitioned_delta_tensor_artifact(
        {
            "sigma": PeftEncoderPartitionDelta(
                partition_name="sigma",
                peft_parameter_deltas={"encoder.q_proj.lora_A": [0.2, 0.0]},
                classifier_head_weight_deltas={
                    "anxiety": [0.1, 0.0],
                    "normal": [-0.1, 0.0],
                },
                classifier_head_bias_deltas={},
            ),
            "psi": PeftEncoderPartitionDelta(
                partition_name="psi",
                peft_parameter_deltas={"encoder.q_proj.lora_A": [0.1, 0.3]},
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
        adapter_kind="peft_classifier",
        backend_name="partitioned_delta_average",
        overrides={"artifact_ref_prefix": "server-aggregate://partitioned_peft"},
        artifact_store=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_peft_state(
            peft_adapter_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/peft_adapter"
            ),
            classifier_head_artifact_ref=artifact_store.ref_for_artifact(
                "rev_000/classifier_head"
            ),
        ),
        update_payloads=(
            make_peft_classifier_delta_payload(
                model_id="tracemind-peft",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_peft_backbone(),
                peft_adapter_config=_peft_adapter_config(),
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

    peft_artifact = _load_peft_state_artifact(
        artifact_store, artifact_ref=result.next_state.peft_adapter_artifact_ref or ""
    )
    assert peft_artifact["encoder.q_proj.lora_A"] == pytest.approx([1.3, 1.3])
    assert result.aggregated_metrics["server_update_partitioned"] == 1.0


def test_peft_classifier_fedavg_two_rounds_accumulates_global_snapshot(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="peft_classifier",
        backend_name="fedavg",
        overrides={"artifact_ref_prefix": "server-aggregate://test_peft"},
        artifact_store=artifact_store,
    )

    first_result = backend.aggregate(
        base_state=_build_peft_state(),
        update_payloads=(
            _build_peft_update(
                peft_deltas={"encoder.q_proj.lora_A": [0.2, 0.4]},
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
            _build_peft_update(
                base_model_revision="rev_001",
                peft_deltas={
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
            _build_peft_update(
                base_model_revision="rev_001",
                peft_deltas={
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

    first_lora = _load_peft_state_artifact(
        artifact_store, artifact_ref=first_result.next_state.peft_adapter_artifact_ref
    )
    second_lora = _load_peft_state_artifact(
        artifact_store,
        artifact_ref=second_result.next_state.peft_adapter_artifact_ref,
    )
    second_applied_lora = _load_applied_peft_delta_artifact(
        artifact_store,
        artifact_ref=second_result.next_state.peft_adapter_artifact_ref,
    )
    first_head = artifact_store.load_json_artifact(
        artifact_ref=first_result.next_state.classifier_head_artifact_ref
    )
    second_head = artifact_store.load_json_artifact(
        artifact_ref=second_result.next_state.classifier_head_artifact_ref
    )

    _assert_vector_mapping_accumulates(
        before=first_lora,
        delta=second_applied_lora,
        after=second_lora,
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


def test_peft_classifier_fedavg_materializes_server_owned_artifact_updates(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    artifact_store.save_json_artifact(
        "u1/peft_adapter_delta",
        {
            "peft_parameter_deltas": {
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
            adapter_kind="peft_classifier",
            method_name="fedavg",
            overrides={"artifact_ref_prefix": "server-aggregate://test_peft"},
        ),
        overrides={"artifact_ref_prefix": "server-aggregate://test_peft"},
        artifact_loader=artifact_store,
    )

    result = backend.aggregate(
        base_state=_build_peft_state(),
        update_payloads=(
            make_peft_classifier_delta_payload(
                model_id="tracemind-peft",
                base_model_revision="rev_000",
                training_scope="adapter_only",
                backbone=_peft_backbone(),
                peft_adapter_config=_peft_adapter_config(),
                label_schema=("anxiety", "normal"),
                example_count=2,
                peft_adapter_delta_artifact_ref=artifact_store.ref_for_artifact(
                    "u1/peft_adapter_delta"
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

    assert result.next_state.peft_adapter_artifact_ref == (
        "server-aggregate://test_peft/rev_001/peft_adapter"
    )
    assert result.aggregated_metrics["client_count"] == 1.0
    assert result.aggregated_metrics["peft_parameter_count"] == 2.0
    assert result.aggregated_metrics["classifier_head_label_count"] == 2.0


def test_aggregation_artifact_store_defaults_to_main_server_state_root() -> None:
    store = AggregationArtifactStore()

    assert store.state_root == (
        Path(__file__).resolve().parents[2] / "state" / "aggregation_artifacts"
    )
    with pytest.raises(ValueError, match="path traversal"):
        store.ref_for_artifact("../escape")


def test_peft_classifier_fedavg_rejects_agent_local_artifact_only_updates() -> None:
    backend = build_shared_adapter_aggregation_backend(
        adapter_kind="peft_classifier",
        backend_name="fedavg",
    )

    with pytest.raises(FileNotFoundError, match="Unsupported aggregation artifact ref"):
        backend.aggregate(
            base_state=_build_peft_state(),
            update_payloads=(
                make_peft_classifier_delta_payload(
                    model_id="tracemind-peft",
                    base_model_revision="rev_000",
                    training_scope="adapter_only",
                    backbone=_peft_backbone(),
                    peft_adapter_config=_peft_adapter_config(),
                    label_schema=("anxiety", "normal"),
                    example_count=1,
                    peft_adapter_delta_artifact_ref=(
                        "agent-local://u1/peft_adapter_delta"
                    ),
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


def _peft_backbone() -> dict[str, str | int]:
    return {
        "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "backbone_revision": "main",
        "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "tokenizer_revision": "main",
        "pooling": "mean",
        "max_length": 256,
        "task_prefix": "",
    }


def _peft_adapter_config() -> dict[str, object]:
    return {
        "peft_adapter_name": "lora",
        "parameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": "all-linear",
            "use_rslora": False,
        },
    }


def _build_peft_state(
    *,
    peft_adapter_artifact_ref: str | None = None,
    classifier_head_artifact_ref: str | None = None,
) -> PeftClassifierState:
    return make_peft_classifier_state_payload(
        model_id="tracemind-peft",
        model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_peft_backbone(),
        peft_adapter_config=_peft_adapter_config(),
        label_schema=("anxiety", "normal"),
        peft_adapter_artifact_ref=peft_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        updated_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def _build_peft_update(
    *,
    base_model_revision: str = "rev_000",
    peft_deltas: dict[str, list[float]],
    head_weight_deltas: dict[str, list[float]],
    head_bias_deltas: dict[str, float],
    example_count: int,
    mean_confidence: float,
) -> PeftClassifierDelta:
    return make_peft_classifier_delta_payload(
        model_id="tracemind-peft",
        base_model_revision=base_model_revision,
        training_scope="adapter_only",
        backbone=_peft_backbone(),
        peft_adapter_config=_peft_adapter_config(),
        label_schema=("anxiety", "normal"),
        example_count=example_count,
        peft_parameter_deltas=peft_deltas,
        classifier_head_weight_deltas=head_weight_deltas,
        classifier_head_bias_deltas=head_bias_deltas,
        delta_format="inline_delta",
        mean_confidence=mean_confidence,
        mean_margin=0.2,
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def _load_peft_state_artifact(
    artifact_store: AggregationArtifactStore,
    *,
    artifact_ref: str,
) -> dict[str, list[float]]:
    tensors, metadata = artifact_store.load_safetensors_artifact(
        artifact_ref=artifact_ref
    )
    return merged_artifacts.parse_peft_adapter_state_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )


def _load_partitioned_peft_state_artifact(
    artifact_store: AggregationArtifactStore,
    *,
    artifact_ref: str,
) -> dict[str, dict[str, list[float]]]:
    tensors, metadata = artifact_store.load_safetensors_artifact(
        artifact_ref=artifact_ref
    )
    return merged_artifacts.parse_partitioned_peft_adapter_state_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )


def _load_applied_peft_delta_artifact(
    artifact_store: AggregationArtifactStore,
    *,
    artifact_ref: str,
) -> dict[str, list[float]]:
    tensors, metadata = artifact_store.load_safetensors_artifact(
        artifact_ref=artifact_ref
    )
    return merged_artifacts.parse_applied_peft_parameter_deltas_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
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
