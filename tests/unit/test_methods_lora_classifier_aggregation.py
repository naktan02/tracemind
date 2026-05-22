"""LoRA-classifier aggregation helper tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import pytest

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_lora_classifier_state,
    materialize_lora_classifier_update,
)
from methods.adaptation.lora_classifier.aggregation.partitioned_state import (
    merge_partitioned_lora_classifier_deltas,
)
from methods.adaptation.lora_classifier.aggregation.state_projection import (
    build_lora_classifier_state_projection,
)
from methods.adaptation.lora_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
    normalize_partition_deltas,
)
from methods.adaptation.lora_classifier.update.partitioned_payload_builder import (
    build_partitioned_delta_payload,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
    LoraClassifierState,
)


class InMemoryJsonArtifactLoader:
    def __init__(self, artifacts: Mapping[str, Mapping[str, object]]) -> None:
        self._artifacts = artifacts

    def load_json_artifact(self, *, artifact_ref: str) -> Mapping[str, object]:
        return self._artifacts[artifact_ref]


def test_materialize_lora_classifier_update_uses_inline_deltas_without_loader() -> None:
    update = _lora_update(
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.2]},
        classifier_head_weight_deltas={
            "anxiety": [0.3, -0.1],
            "normal": [-0.3, 0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.05},
        delta_l2_norm=None,
    )

    materialized = materialize_lora_classifier_update(
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


def test_materialize_lora_classifier_update_reads_server_artifact_refs() -> None:
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

    materialized = materialize_lora_classifier_update(
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


def test_materialize_base_lora_classifier_state_reads_global_snapshot_artifacts() -> (
    None
):
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

    materialized = materialize_base_lora_classifier_state(
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


def test_lora_classifier_state_projection_applies_delta_to_base_snapshot() -> None:
    projection = build_lora_classifier_state_projection(
        base_state=_lora_state(),
        base_parameters=LoraClassifierMaterializedState(
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


def test_lora_classifier_state_projection_rejects_delta_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="delta dimension mismatch"):
        build_lora_classifier_state_projection(
            base_state=_lora_state(),
            base_parameters=LoraClassifierMaterializedState(
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


def test_lora_classifier_partitioned_deltas_merge_without_fedmatch_names() -> None:
    partitions = normalize_partition_deltas(
        (
            LoraClassifierPartitionDelta(
                partition_name="private",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.2, -0.1]},
                classifier_head_weight_deltas={"anxiety": [0.1, 0.3]},
                classifier_head_bias_deltas={"anxiety": 0.05},
            ),
            LoraClassifierPartitionDelta(
                partition_name="shared",
                lora_parameter_deltas={"encoder.q_proj.lora_A": [0.4, 0.5]},
                classifier_head_weight_deltas={"anxiety": [0.2, -0.1]},
                classifier_head_bias_deltas={"anxiety": 0.15},
            ),
        )
    )

    merged = merge_partitioned_lora_classifier_deltas(partitions)

    assert merged.partition_name == "merged"
    assert merged.lora_parameter_deltas["encoder.q_proj.lora_A"] == pytest.approx(
        [0.6, 0.4]
    )
    assert merged.classifier_head_weight_deltas["anxiety"] == pytest.approx([0.3, 0.2])
    assert merged.classifier_head_bias_deltas["anxiety"] == pytest.approx(0.2)


def test_lora_classifier_partitioned_payload_keeps_partition_names() -> None:
    payload = build_partitioned_delta_payload(
        (
            LoraClassifierPartitionDelta(
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
