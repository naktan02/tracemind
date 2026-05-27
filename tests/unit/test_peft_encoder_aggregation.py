"""PEFT encoder aggregation helper tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import pytest

from methods.adaptation.peft_text_classifier.aggregation import (
    peft_encoder_partitioned_projection as peft_part_projection,
)
from methods.adaptation.peft_text_classifier.aggregation import (
    peft_encoder_partitioned_state as peft_part_state,
)
from methods.adaptation.peft_text_classifier.aggregation import (
    peft_encoder_state_projection as peft_state_projection,
)
from methods.adaptation.peft_text_classifier.update import (
    merged_tensor_artifact as merged_artifacts,
)
from methods.adaptation.peft_text_classifier.update import (
    partitioned_payload_builder as partitioned_payloads,
)
from methods.adaptation.peft_text_classifier.update import (
    partitioned_tensor_artifact as partitioned_artifacts,
)
from methods.adaptation.peft_text_classifier.update.materialization import (
    PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY,
    PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    PARTITIONED_LORA_STATE_PARAMETERS_KEY,
    PARTITIONED_PEFT_STATE_PARAMETERS_KEY,
    PEFT_STATE_PARAMETERS_KEY,
    PeftEncoderMaterializedState,
    materialize_base_peft_encoder_partitioned_state,
    materialize_base_peft_encoder_state,
    materialize_peft_encoder_partitioned_update,
    materialize_peft_encoder_update,
)
from methods.adaptation.peft_text_classifier.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
    normalize_partition_deltas,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierDelta,
    PeftClassifierState,
)


class InMemoryJsonArtifactLoader:
    def __init__(self, artifacts: Mapping[str, Mapping[str, object]]) -> None:
        self._artifacts = artifacts

    def load_json_artifact(self, *, artifact_ref: str) -> Mapping[str, object]:
        return self._artifacts[artifact_ref]


class InMemoryTensorArtifactLoader(InMemoryJsonArtifactLoader):
    def __init__(
        self,
        artifacts: Mapping[str, Mapping[str, object]],
        tensor_artifacts: Mapping[str, tuple[Mapping[str, object], Mapping[str, str]]],
    ) -> None:
        super().__init__(artifacts)
        self._tensor_artifacts = tensor_artifacts

    def load_safetensors_artifact(
        self,
        *,
        artifact_ref: str,
    ) -> tuple[Mapping[str, object], Mapping[str, str]]:
        return self._tensor_artifacts[artifact_ref]


def test_materialize_peft_encoder_update_uses_inline_deltas_without_loader() -> None:
    update = _lora_update(
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05},
        delta_l2_norm=None,
    )

    materialized = materialize_peft_encoder_update(
        payload=update,
        context=_aggregation_context(),
    )

    assert materialized.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.1, -0.2]
    )
    assert materialized.classifier_head_weight_deltas["anxiety"] == pytest.approx(
        [0.3, -0.1]
    )
    assert materialized.classifier_head_bias_deltas["anxiety"] == pytest.approx(0.05)
    assert materialized.classifier_head_bias_deltas["normal"] == pytest.approx(0.0)
    assert materialized.delta_l2_norm == pytest.approx(
        (0.1**2 + 0.2**2 + 0.3**2 + 0.1**2 + 0.3**2 + 0.1**2 + 0.05**2) ** 0.5
    )


def test_materialize_peft_encoder_update_reads_server_artifact_refs() -> None:
    loader = InMemoryJsonArtifactLoader(
        {
            "aggregation_artifact://client/lora_delta": {
                "lora_parameter_deltas": {
                    "encoder.q_proj.lora_A": [0.2, 0.4],
                },
            },
            "aggregation_artifact://client/head_delta": {
                "classifier_head_weight_deltas": {
                    "anxiety": [0.5, -0.1],
                    "normal": [-0.5, 0.1],
                },
                "classifier_head_bias_deltas": {
                    "anxiety": 0.2,
                    "normal": -0.2,
                },
            },
        }
    )
    update = _lora_update(
        lora_delta_artifact_ref="aggregation_artifact://client/lora_delta",
        classifier_head_delta_artifact_ref="aggregation_artifact://client/head_delta",
        lora_parameter_deltas=None,
        classifier_head_weight_deltas=None,
        classifier_head_bias_deltas={},
        delta_format="server_uploaded_artifact_ref",
    )

    materialized = materialize_peft_encoder_update(
        payload=update,
        context=_aggregation_context(loader=loader),
    )

    assert materialized.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.2, 0.4]
    )
    assert materialized.classifier_head_weight_deltas["normal"] == pytest.approx(
        [-0.5, 0.1]
    )
    assert materialized.classifier_head_bias_deltas == pytest.approx(
        {"anxiety": 0.2, "normal": -0.2}
    )


def test_materialize_peft_classifier_update_reads_v2_fields() -> None:
    loader = InMemoryJsonArtifactLoader(
        {
            "aggregation_artifact://client/peft_delta": {
                "peft_parameter_deltas": {
                    "encoder.q_proj.lora_A": [0.2, 0.4],
                },
            },
            "aggregation_artifact://client/head_delta": {
                "classifier_head_weight_deltas": {
                    "anxiety": [0.5, -0.1],
                    "normal": [-0.5, 0.1],
                },
                "classifier_head_bias_deltas": {
                    "anxiety": 0.2,
                    "normal": -0.2,
                },
            },
        }
    )
    update = _peft_update(
        peft_adapter_delta_artifact_ref="aggregation_artifact://client/peft_delta",
        classifier_head_delta_artifact_ref="aggregation_artifact://client/head_delta",
        peft_parameter_deltas=None,
        classifier_head_weight_deltas=None,
        classifier_head_bias_deltas={},
        delta_format="server_uploaded_artifact_ref",
    )

    materialized = materialize_peft_encoder_update(
        payload=update,
        context=_aggregation_context(loader=loader),
    )

    assert materialized.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.2, 0.4]
    )
    assert materialized.classifier_head_weight_deltas["normal"] == pytest.approx(
        [-0.5, 0.1]
    )
    assert materialized.classifier_head_bias_deltas == pytest.approx(
        {"anxiety": 0.2, "normal": -0.2}
    )


def test_merged_delta_tensor_artifacts_roundtrip() -> None:
    lora_tensors, lora_metadata = merged_artifacts.build_lora_delta_tensor_artifact(
        {
            "encoder.q_proj.lora_A": [0.2, -0.1],
        }
    )
    head_tensors, head_metadata = (
        merged_artifacts.build_classifier_head_delta_tensor_artifact(
            classifier_head_weight_deltas={
                "anxiety": [0.5, -0.1],
                "normal": [-0.5, 0.1],
            },
            classifier_head_bias_deltas={
                "anxiety": 0.2,
                "normal": -0.2,
            },
        )
    )

    lora_deltas = merged_artifacts.parse_lora_delta_tensor_artifact(
        tensors=lora_tensors,
        metadata=lora_metadata,
    )
    head_weight_deltas, head_bias_deltas = (
        merged_artifacts.parse_classifier_head_delta_tensor_artifact(
            tensors=head_tensors,
            metadata=head_metadata,
        )
    )

    assert lora_deltas["encoder.q_proj.lora_A"] == pytest.approx([0.2, -0.1])
    assert head_weight_deltas["normal"] == pytest.approx([-0.5, 0.1])
    assert head_bias_deltas == pytest.approx({"anxiety": 0.2, "normal": -0.2})


def test_materialize_peft_encoder_update_reads_server_tensor_artifact_refs() -> None:
    lora_tensors, lora_metadata = merged_artifacts.build_lora_delta_tensor_artifact(
        {
            "encoder.q_proj.lora_A": [0.2, 0.4],
        }
    )
    head_tensors, head_metadata = (
        merged_artifacts.build_classifier_head_delta_tensor_artifact(
            classifier_head_weight_deltas={
                "anxiety": [0.5, -0.1],
                "normal": [-0.5, 0.1],
            },
            classifier_head_bias_deltas={
                "anxiety": 0.2,
                "normal": -0.2,
            },
        )
    )
    loader = InMemoryTensorArtifactLoader(
        artifacts={},
        tensor_artifacts={
            "aggregation_artifact://client/lora_delta": (
                lora_tensors,
                lora_metadata,
            ),
            "aggregation_artifact://client/head_delta": (
                head_tensors,
                head_metadata,
            ),
        },
    )
    update = _lora_update(
        lora_delta_artifact_ref="aggregation_artifact://client/lora_delta",
        classifier_head_delta_artifact_ref="aggregation_artifact://client/head_delta",
        lora_parameter_deltas=None,
        classifier_head_weight_deltas=None,
        classifier_head_bias_deltas={},
        delta_format="server_uploaded_artifact_ref",
    )

    materialized = materialize_peft_encoder_update(
        payload=update,
        context=_aggregation_context(loader=loader),
    )

    assert materialized.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.2, 0.4]
    )
    assert materialized.classifier_head_weight_deltas["normal"] == pytest.approx(
        [-0.5, 0.1]
    )
    assert materialized.classifier_head_bias_deltas == pytest.approx(
        {"anxiety": 0.2, "normal": -0.2}
    )


def test_materialize_base_peft_encoder_state_reads_global_snapshot_artifacts() -> None:
    loader = InMemoryJsonArtifactLoader(
        {
            "server-aggregate://rev_000/lora_adapter": {
                "lora_parameters": {
                    "encoder.q_proj.lora_A": [1.0, 2.0],
                },
            },
            "server-aggregate://rev_000/classifier_head": {
                "classifier_head_weights": {
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                "classifier_head_biases": {
                    "anxiety": 0.3,
                    "normal": -0.3,
                },
            },
        }
    )

    materialized = materialize_base_peft_encoder_state(
        base_state=_lora_state(
            lora_adapter_artifact_ref="server-aggregate://rev_000/lora_adapter",
            classifier_head_artifact_ref="server-aggregate://rev_000/classifier_head",
        ),
        context=_aggregation_context(loader=loader),
    )

    assert materialized.lora_parameters["encoder.q_proj.lora_A"] == pytest.approx(
        [1.0, 2.0]
    )
    assert materialized.classifier_head_weights["anxiety"] == pytest.approx([0.1, 0.2])
    assert materialized.classifier_head_biases == pytest.approx(
        {"anxiety": 0.3, "normal": -0.3}
    )


def test_materialize_base_peft_classifier_state_reads_v2_artifacts() -> None:
    loader = InMemoryJsonArtifactLoader(
        {
            "server-aggregate://rev_000/peft_adapter": {
                PEFT_STATE_PARAMETERS_KEY: {
                    "encoder.q_proj.lora_A": [1.0, 2.0],
                },
            },
            "server-aggregate://rev_000/classifier_head": {
                "classifier_head_weights": {
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                "classifier_head_biases": {
                    "anxiety": 0.3,
                    "normal": -0.3,
                },
            },
        }
    )

    materialized = materialize_base_peft_encoder_state(
        base_state=_peft_state(
            peft_adapter_artifact_ref="server-aggregate://rev_000/peft_adapter",
            classifier_head_artifact_ref="server-aggregate://rev_000/classifier_head",
        ),
        context=_aggregation_context(loader=loader),
    )

    assert materialized.lora_parameters["encoder.q_proj.lora_A"] == pytest.approx(
        [1.0, 2.0]
    )
    assert materialized.classifier_head_weights["anxiety"] == pytest.approx([0.1, 0.2])
    assert materialized.classifier_head_biases == pytest.approx(
        {"anxiety": 0.3, "normal": -0.3}
    )


def test_peft_encoder_state_projection_applies_delta_to_base_snapshot() -> None:
    projection = peft_state_projection.build_peft_encoder_state_projection(
        base_state=_lora_state(),
        base_parameters=PeftEncoderMaterializedState(
            lora_parameters={
                "encoder.q_proj.lora_A": [1.0, 2.0],
                "encoder.extra.lora_A": [0.5],
            },
            classifier_head_weights={
                "anxiety": [0.1, 0.2],
                "normal": [-0.1, -0.2],
            },
            classifier_head_biases={"anxiety": 0.3},
        ),
        next_model_revision="rev_001",
        updated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
        lora_adapter_artifact_ref="server-aggregate://rev_001/lora_adapter",
        classifier_head_artifact_ref="server-aggregate://rev_001/classifier_head",
        artifact_format="server_aggregated_artifact_ref",
        lora_parameter_deltas={
            "encoder.q_proj.lora_A": [0.2, -0.4],
            "encoder.q_proj.lora_B": [0.7],
        },
        classifier_head_weight_deltas={
            "anxiety": [0.5, -0.1],
            "normal": [-0.5, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.2, "normal": -0.2},
    )

    assert projection.next_state.model_revision == "rev_001"
    assert projection.next_state.lora_adapter_artifact_ref == (
        "server-aggregate://rev_001/lora_adapter"
    )
    assert projection.artifacts["server-aggregate://rev_001/lora_adapter"][
        "lora_parameters"
    ]["encoder.q_proj.lora_A"] == pytest.approx([1.2, 1.6])
    assert projection.artifacts["server-aggregate://rev_001/lora_adapter"][
        "lora_parameters"
    ]["encoder.q_proj.lora_B"] == pytest.approx([0.7])
    assert projection.artifacts["server-aggregate://rev_001/classifier_head"][
        "classifier_head_weights"
    ]["normal"] == pytest.approx([-0.6, -0.1])
    assert projection.artifacts["server-aggregate://rev_001/classifier_head"][
        "classifier_head_biases"
    ] == pytest.approx({"anxiety": 0.5, "normal": -0.2})


def test_peft_classifier_state_projection_uses_v2_state_and_artifact_keys() -> None:
    projection = peft_state_projection.build_peft_encoder_state_projection(
        base_state=_peft_state(),
        base_parameters=PeftEncoderMaterializedState(
            lora_parameters={"encoder.q_proj.lora_A": [1.0, 2.0]},
            classifier_head_weights={
                "anxiety": [0.1, 0.2],
                "normal": [-0.1, -0.2],
            },
            classifier_head_biases={"anxiety": 0.3},
        ),
        next_model_revision="rev_001",
        updated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
        lora_adapter_artifact_ref="server-aggregate://rev_001/peft_adapter",
        classifier_head_artifact_ref="server-aggregate://rev_001/classifier_head",
        artifact_format="server_aggregated_artifact_ref",
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.2, -0.4]},
        classifier_head_weight_deltas={
            "anxiety": [0.5, -0.1],
            "normal": [-0.5, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.2, "normal": -0.2},
    )

    assert isinstance(projection.next_state, PeftClassifierState)
    assert projection.next_state.peft_adapter_artifact_ref == (
        "server-aggregate://rev_001/peft_adapter"
    )
    assert projection.artifacts["server-aggregate://rev_001/peft_adapter"][
        PEFT_STATE_PARAMETERS_KEY
    ]["encoder.q_proj.lora_A"] == pytest.approx([1.2, 1.6])
    assert (
        "applied_peft_parameter_deltas"
        in projection.artifacts["server-aggregate://rev_001/peft_adapter"]
    )


def test_peft_encoder_state_projection_rejects_delta_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="delta dimension mismatch"):
        peft_state_projection.build_peft_encoder_state_projection(
            base_state=_lora_state(),
            base_parameters=PeftEncoderMaterializedState(
                lora_parameters={"encoder.q_proj.lora_A": [1.0, 2.0]},
                classifier_head_weights={
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                classifier_head_biases={},
            ),
            next_model_revision="rev_001",
            updated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
            lora_adapter_artifact_ref="server-aggregate://rev_001/lora_adapter",
            classifier_head_artifact_ref="server-aggregate://rev_001/classifier_head",
            artifact_format="server_aggregated_artifact_ref",
            lora_parameter_deltas={"encoder.q_proj.lora_A": [0.2]},
            classifier_head_weight_deltas={
                "anxiety": [0.5, -0.1],
                "normal": [-0.5, 0.1],
            },
            classifier_head_bias_deltas={},
        )


def test_peft_encoder_partitioned_deltas_merge_without_fedmatch_names() -> None:
    partitions = normalize_partition_deltas(
        (
            PeftEncoderPartitionDelta(
                partition_name="private",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.2, -0.1]},
                classifier_head_weight_deltas={"anxiety": [0.1, 0.3]},
                classifier_head_bias_deltas={"anxiety": 0.05},
            ),
            PeftEncoderPartitionDelta(
                partition_name="shared",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.4, 0.5]},
                classifier_head_weight_deltas={"anxiety": [0.2, -0.1]},
                classifier_head_bias_deltas={"anxiety": 0.15},
            ),
        )
    )

    merged = peft_part_state.merge_partitioned_peft_encoder_deltas(partitions)

    assert merged.partition_name == "merged"
    assert merged.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.6, 0.4]
    )
    assert merged.classifier_head_weight_deltas["anxiety"] == pytest.approx([0.3, 0.2])
    assert merged.classifier_head_bias_deltas["anxiety"] == pytest.approx(0.2)


def test_peft_encoder_partition_delta_applies_to_materialized_state() -> None:
    base = PeftEncoderMaterializedState(
        lora_parameters={"encoder_lora.weight": [0.1, 0.2]},
        classifier_head_weights={"anxiety": [0.3, 0.4]},
        classifier_head_biases={"anxiety": 0.5},
    )
    delta = PeftEncoderPartitionDelta(
        partition_name="merged",
        lora_parameter_deltas={"encoder_lora.weight": [0.2, -0.1]},
        classifier_head_weight_deltas={
            "anxiety": [0.1, 0.1],
            "normal": [-0.2, 0.2],
        },
        classifier_head_bias_deltas={"anxiety": -0.1, "normal": 0.2},
    )

    state = peft_part_state.apply_peft_encoder_partition_delta_to_state(
        base_parameters=base,
        delta=delta,
    )

    assert state.lora_parameters["encoder_lora.weight"] == pytest.approx([0.3, 0.1])
    assert state.classifier_head_weights["anxiety"] == pytest.approx([0.4, 0.5])
    assert state.classifier_head_weights["normal"] == pytest.approx([-0.2, 0.2])
    assert state.classifier_head_biases == pytest.approx(
        {"anxiety": 0.4, "normal": 0.2}
    )


def test_peft_encoder_state_splits_published_state_by_residual_factor() -> None:
    published = PeftEncoderMaterializedState(
        lora_parameters={"encoder_lora.weight": [1.2, -2.4]},
        classifier_head_weights={"anxiety": [3.6, -4.8]},
        classifier_head_biases={"anxiety": 6.0},
    )

    partitions = peft_part_state.split_peft_encoder_state_by_residual_factor(
        published_parameters=published,
        base_partition_name="sigma",
        residual_partition_name="psi",
        residual_factor=0.2,
    )

    assert partitions["sigma"].lora_parameters["encoder_lora.weight"] == (
        pytest.approx([1.0, -2.0])
    )
    assert partitions["psi"].lora_parameters["encoder_lora.weight"] == pytest.approx(
        [0.2, -0.4]
    )
    assert partitions["sigma"].classifier_head_weights["anxiety"] == pytest.approx(
        [3.0, -4.0]
    )
    assert partitions["psi"].classifier_head_weights["anxiety"] == pytest.approx(
        [0.6, -0.8]
    )
    assert partitions["sigma"].classifier_head_biases["anxiety"] == pytest.approx(5.0)
    assert partitions["psi"].classifier_head_biases["anxiety"] == pytest.approx(1.0)


def test_peft_encoder_partition_delta_rejects_state_dimension_mismatch() -> None:
    base = PeftEncoderMaterializedState(
        lora_parameters={"encoder_lora.weight": [0.1, 0.2]},
        classifier_head_weights={},
        classifier_head_biases={},
    )
    delta = PeftEncoderPartitionDelta(
        partition_name="merged",
        lora_parameter_deltas={"encoder_lora.weight": [0.2]},
        classifier_head_weight_deltas={},
        classifier_head_bias_deltas={},
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        peft_part_state.apply_peft_encoder_partition_delta_to_state(
            base_parameters=base,
            delta=delta,
        )


def test_materialize_lora_classifier_partitioned_base_state_reads_artifact_metadata():
    state = materialize_base_peft_encoder_partitioned_state(
        base_state=_lora_state(
            lora_adapter_artifact_ref="aggregation_artifact://state/lora",
            classifier_head_artifact_ref="aggregation_artifact://state/head",
        ),
        context=_aggregation_context(
            loader=InMemoryJsonArtifactLoader(
                {
                    "aggregation_artifact://state/lora": {
                        PARTITIONED_LORA_STATE_PARAMETERS_KEY: {
                            "sigma": {"encoder_lora.weight": [0.1, 0.2]},
                            "psi": {"encoder_lora.weight": [0.3, 0.4]},
                        }
                    },
                    "aggregation_artifact://state/head": {
                        PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY: {
                            "sigma": {"anxiety": [0.5, 0.6]},
                        },
                        PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY: {
                            "psi": {"anxiety": -0.1},
                        },
                    },
                }
            )
        ),
    )

    assert set(state) == {"sigma", "psi"}
    assert state["sigma"].lora_parameters["encoder_lora.weight"] == pytest.approx(
        [0.1, 0.2]
    )
    assert state["sigma"].classifier_head_weights["anxiety"] == pytest.approx(
        [0.5, 0.6]
    )
    assert state["psi"].lora_parameters["encoder_lora.weight"] == pytest.approx(
        [0.3, 0.4]
    )
    assert state["psi"].classifier_head_biases["anxiety"] == pytest.approx(-0.1)


def test_materialize_peft_classifier_partitioned_base_state_reads_v2_metadata():
    state = materialize_base_peft_encoder_partitioned_state(
        base_state=_peft_state(
            peft_adapter_artifact_ref="aggregation_artifact://state/peft",
            classifier_head_artifact_ref="aggregation_artifact://state/head",
        ),
        context=_aggregation_context(
            loader=InMemoryJsonArtifactLoader(
                {
                    "aggregation_artifact://state/peft": {
                        PARTITIONED_PEFT_STATE_PARAMETERS_KEY: {
                            "sigma": {"encoder_lora.weight": [0.1, 0.2]},
                            "psi": {"encoder_lora.weight": [0.3, 0.4]},
                        }
                    },
                    "aggregation_artifact://state/head": {
                        PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY: {
                            "sigma": {"anxiety": [0.5, 0.6]},
                        },
                        PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY: {
                            "psi": {"anxiety": -0.1},
                        },
                    },
                }
            )
        ),
    )

    assert set(state) == {"sigma", "psi"}
    assert state["sigma"].lora_parameters["encoder_lora.weight"] == pytest.approx(
        [0.1, 0.2]
    )
    assert state["psi"].lora_parameters["encoder_lora.weight"] == pytest.approx(
        [0.3, 0.4]
    )


def test_peft_encoder_partitioned_payload_keeps_partition_names() -> None:
    payload = partitioned_payloads.build_partitioned_delta_payload(
        (
            PeftEncoderPartitionDelta(
                partition_name="shared",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.4]},
            ),
        )
    )

    assert payload == {
        "partitions": {
            "shared": {
                "lora_parameter_deltas": {"encoder.q_proj.lora_A": [0.4]},
                "classifier_head_weight_deltas": {},
                "classifier_head_bias_deltas": {},
            }
        }
    }


def test_materialize_peft_encoder_partitioned_update_reads_shared_payload() -> None:
    update = _lora_update(
        lora_parameter_deltas=None,
        classifier_head_weight_deltas=None,
        classifier_head_bias_deltas={},
        partitioned_deltas={
            "sigma": {
                "lora_parameter_deltas": {"encoder.q_proj.lora_A": [0.2]},
                "classifier_head_weight_deltas": {
                    "anxiety": [0.1, 0.0],
                    "normal": [-0.1, 0.0],
                },
                "classifier_head_bias_deltas": {"anxiety": 0.05},
            },
            "psi": {
                "lora_parameter_deltas": {"encoder.q_proj.lora_A": [0.3]},
                "classifier_head_weight_deltas": {
                    "anxiety": [0.2, 0.0],
                    "normal": [-0.2, 0.0],
                },
                "classifier_head_bias_deltas": {"normal": -0.04},
            },
        },
        delta_format="partitioned_update",
        delta_l2_norm=None,
    )

    partitions = materialize_peft_encoder_partitioned_update(payload=update)

    assert set(partitions) == {"sigma", "psi"}
    assert partitions["sigma"].lora_parameter_deltas[
        "encoder.q_proj.lora_A"
    ] == pytest.approx([0.2])
    assert partitions["psi"].classifier_head_bias_deltas == pytest.approx(
        {"anxiety": 0.0, "normal": -0.04}
    )


def test_materialize_peft_classifier_partitioned_update_reads_v2_payload() -> None:
    update = _peft_update(
        peft_parameter_deltas=None,
        classifier_head_weight_deltas=None,
        classifier_head_bias_deltas={},
        partitioned_deltas={
            "sigma": {
                "peft_parameter_deltas": {"encoder.q_proj.lora_A": [0.2]},
                "classifier_head_weight_deltas": {
                    "anxiety": [0.1, 0.0],
                    "normal": [-0.1, 0.0],
                },
                "classifier_head_bias_deltas": {"anxiety": 0.05},
            },
            "psi": {
                "peft_parameter_deltas": {"encoder.q_proj.lora_A": [0.3]},
                "classifier_head_weight_deltas": {
                    "anxiety": [0.2, 0.0],
                    "normal": [-0.2, 0.0],
                },
                "classifier_head_bias_deltas": {"normal": -0.04},
            },
        },
        delta_format="partitioned_update",
        delta_l2_norm=None,
    )

    partitions = materialize_peft_encoder_partitioned_update(payload=update)

    assert set(partitions) == {"sigma", "psi"}
    assert partitions["sigma"].lora_parameter_deltas[
        "encoder.q_proj.lora_A"
    ] == pytest.approx([0.2])
    assert partitions["psi"].classifier_head_bias_deltas == pytest.approx(
        {"anxiety": 0.0, "normal": -0.04}
    )


def test_materialize_peft_encoder_partitioned_update_reads_artifact_ref() -> None:
    loader = InMemoryJsonArtifactLoader(
        {
            "aggregation_artifact://client/partitioned_delta": {
                "partitions": {
                    "sigma": {
                        "lora_parameter_deltas": {
                            "encoder.q_proj.lora_A": [0.2],
                        },
                        "classifier_head_weight_deltas": {
                            "anxiety": [0.1, 0.0],
                            "normal": [-0.1, 0.0],
                        },
                        "classifier_head_bias_deltas": {"anxiety": 0.05},
                    },
                    "psi": {
                        "lora_parameter_deltas": {
                            "encoder.q_proj.lora_A": [0.3],
                        },
                        "classifier_head_weight_deltas": {
                            "anxiety": [0.2, 0.0],
                            "normal": [-0.2, 0.0],
                        },
                        "classifier_head_bias_deltas": {"normal": -0.04},
                    },
                }
            }
        }
    )
    update = _lora_update(
        lora_parameter_deltas=None,
        classifier_head_weight_deltas=None,
        classifier_head_bias_deltas={},
        partitioned_deltas_artifact_ref=(
            "aggregation_artifact://client/partitioned_delta"
        ),
        delta_format="server_uploaded_artifact_ref",
        delta_l2_norm=1.5,
    )

    partitions = materialize_peft_encoder_partitioned_update(
        payload=update,
        context=_aggregation_context(loader=loader),
    )

    assert set(partitions) == {"sigma", "psi"}
    assert partitions["sigma"].lora_parameter_deltas[
        "encoder.q_proj.lora_A"
    ] == pytest.approx([0.2])
    assert partitions["psi"].classifier_head_bias_deltas == pytest.approx(
        {"normal": -0.04}
    )


def test_partitioned_delta_tensor_artifact_roundtrips() -> None:
    tensors, metadata = partitioned_artifacts.build_partitioned_delta_tensor_artifact(
        {
            "sigma": PeftEncoderPartitionDelta(
                partition_name="sigma",
                lora_parameter_deltas={
                    "encoder.q_proj.lora_A": [0.2, -0.1],
                },
                classifier_head_weight_deltas={
                    "anxiety": [0.1, 0.0],
                },
                classifier_head_bias_deltas={"anxiety": 0.05},
            )
        }
    )

    partitions = partitioned_artifacts.parse_partitioned_delta_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )

    assert set(partitions) == {"sigma"}
    assert partitions["sigma"].lora_parameter_deltas[
        "encoder.q_proj.lora_A"
    ] == pytest.approx([0.2, -0.1])
    assert partitions["sigma"].classifier_head_weight_deltas["anxiety"] == (
        pytest.approx([0.1, 0.0])
    )
    assert partitions["sigma"].classifier_head_bias_deltas == pytest.approx(
        {"anxiety": 0.05}
    )


def test_materialize_peft_encoder_partitioned_update_reads_tensor_artifact_ref() -> (
    None
):
    tensors, metadata = partitioned_artifacts.build_partitioned_delta_tensor_artifact(
        {
            "psi": PeftEncoderPartitionDelta(
                partition_name="psi",
                lora_parameter_deltas={
                    "encoder.q_proj.lora_A": [0.3],
                },
                classifier_head_weight_deltas={
                    "anxiety": [0.2, 0.0],
                    "normal": [-0.2, 0.0],
                },
                classifier_head_bias_deltas={"normal": -0.04},
            )
        }
    )
    loader = InMemoryTensorArtifactLoader(
        artifacts={},
        tensor_artifacts={
            "aggregation_artifact://client/partitioned_delta": (tensors, metadata)
        },
    )
    update = _lora_update(
        lora_parameter_deltas=None,
        classifier_head_weight_deltas=None,
        classifier_head_bias_deltas={},
        partitioned_deltas_artifact_ref=(
            "aggregation_artifact://client/partitioned_delta"
        ),
        delta_format="server_uploaded_artifact_ref",
        delta_l2_norm=1.5,
    )

    partitions = materialize_peft_encoder_partitioned_update(
        payload=update,
        context=_aggregation_context(loader=loader),
    )

    assert set(partitions) == {"psi"}
    assert partitions["psi"].lora_parameter_deltas[
        "encoder.q_proj.lora_A"
    ] == pytest.approx([0.3])
    assert partitions["psi"].classifier_head_bias_deltas == pytest.approx(
        {"normal": -0.04}
    )


def test_peft_encoder_partitioned_delta_average_merges_partitions_per_client() -> None:
    result = peft_part_projection.compute_peft_encoder_partitioned_delta_average(
        label_schema=("anxiety", "normal"),
        updates=(
            peft_part_projection.PeftEncoderPartitionedDeltaAverageUpdate(
                partitions={
                    "sigma": PeftEncoderPartitionDelta(
                        partition_name="sigma",
                        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.2, 0.0]},
                        classifier_head_weight_deltas={
                            "anxiety": [0.1, 0.0],
                            "normal": [-0.1, 0.0],
                        },
                    ),
                    "psi": PeftEncoderPartitionDelta(
                        partition_name="psi",
                        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, 0.3]},
                        classifier_head_weight_deltas={
                            "anxiety": [0.2, 0.2],
                            "normal": [-0.2, -0.2],
                        },
                        classifier_head_bias_deltas={"anxiety": 0.03},
                    ),
                },
                example_count=2,
                mean_confidence=0.9,
                mean_margin=0.3,
                delta_l2_norm=0.5,
            ),
            peft_part_projection.PeftEncoderPartitionedDeltaAverageUpdate(
                partitions={
                    "sigma": PeftEncoderPartitionDelta(
                        partition_name="sigma",
                        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.0, 0.6]},
                        classifier_head_weight_deltas={
                            "anxiety": [0.0, 0.3],
                            "normal": [0.0, -0.3],
                        },
                        classifier_head_bias_deltas={"normal": -0.06},
                    )
                },
                example_count=1,
                mean_confidence=0.6,
                mean_margin=0.1,
                delta_l2_norm=0.2,
            ),
        ),
    )

    assert result.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.2, 0.4]
    )
    assert result.classifier_head_weight_deltas["anxiety"] == pytest.approx(
        [0.2, 0.23333333333333334]
    )
    assert result.classifier_head_bias_deltas["anxiety"] == pytest.approx(0.02)
    assert result.classifier_head_bias_deltas["normal"] == pytest.approx(-0.02)
    assert result.aggregated_metrics["server_update_partitioned"] == 1.0
    assert result.aggregated_metrics["partition_count_total"] == 3.0
    assert result.update_count == 2


def _aggregation_context(
    *,
    loader: InMemoryJsonArtifactLoader | None = None,
) -> FederatedAggregationContext:
    return FederatedAggregationContext(
        next_model_revision="rev_001",
        aggregated_at=datetime(2026, 4, 8, 1, tzinfo=timezone.utc),
        artifact_loader=loader,
    )


def _lora_state(
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


def _lora_update(
    *,
    lora_parameter_deltas: dict[str, list[float]] | None = None,
    classifier_head_weight_deltas: dict[str, list[float]] | None = None,
    classifier_head_bias_deltas: dict[str, float] | None = None,
    partitioned_deltas: dict[str, dict[str, object]] | None = None,
    partitioned_deltas_artifact_ref: str | None = None,
    lora_delta_artifact_ref: str | None = None,
    classifier_head_delta_artifact_ref: str | None = None,
    delta_l2_norm: float | None = 1.0,
    delta_format: str = "inline_delta",
) -> LoraClassifierDelta:
    return make_lora_classifier_delta_payload(
        model_id="tracemind-lora",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_lora_backbone(),
        lora_config=_lora_config(),
        label_schema=("anxiety", "normal"),
        example_count=2,
        lora_delta_artifact_ref=lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=classifier_head_delta_artifact_ref,
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
        partitioned_deltas=partitioned_deltas,
        partitioned_deltas_artifact_ref=partitioned_deltas_artifact_ref,
        delta_format=delta_format,
        mean_confidence=0.9,
        mean_margin=0.2,
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        delta_l2_norm=delta_l2_norm,
    )


def _peft_state(
    *,
    peft_adapter_artifact_ref: str | None = None,
    classifier_head_artifact_ref: str | None = None,
) -> PeftClassifierState:
    return make_peft_classifier_state_payload(
        model_id="tracemind-lora",
        model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_lora_backbone(),
        peft_adapter_config=_peft_config(),
        label_schema=("anxiety", "normal"),
        peft_adapter_artifact_ref=peft_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        updated_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def _peft_update(
    *,
    peft_parameter_deltas: dict[str, list[float]] | None = None,
    classifier_head_weight_deltas: dict[str, list[float]] | None = None,
    classifier_head_bias_deltas: dict[str, float] | None = None,
    partitioned_deltas: dict[str, dict[str, object]] | None = None,
    partitioned_deltas_artifact_ref: str | None = None,
    peft_adapter_delta_artifact_ref: str | None = None,
    classifier_head_delta_artifact_ref: str | None = None,
    delta_l2_norm: float | None = 1.0,
    delta_format: str = "inline_delta",
) -> PeftClassifierDelta:
    return make_peft_classifier_delta_payload(
        model_id="tracemind-lora",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_lora_backbone(),
        peft_adapter_config=_peft_config(),
        label_schema=("anxiety", "normal"),
        example_count=2,
        peft_adapter_delta_artifact_ref=peft_adapter_delta_artifact_ref,
        classifier_head_delta_artifact_ref=classifier_head_delta_artifact_ref,
        peft_parameter_deltas=peft_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
        partitioned_deltas=partitioned_deltas,
        partitioned_deltas_artifact_ref=partitioned_deltas_artifact_ref,
        delta_format=delta_format,
        mean_confidence=0.9,
        mean_margin=0.2,
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        delta_l2_norm=delta_l2_norm,
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


def _peft_config() -> dict[str, object]:
    return {
        "peft_adapter_name": "lora",
        "parameters": _lora_config(),
    }
