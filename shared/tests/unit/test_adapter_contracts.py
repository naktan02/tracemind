from __future__ import annotations

from pathlib import Path

import pytest

from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadAdapterStatePayload,
    ClassifierHeadAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.io import (
    dump_shared_adapter_state_payload,
    dump_shared_adapter_update_payload,
    load_shared_adapter_state_payload,
    load_shared_adapter_update_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    LoraClassifierAdapterStatePayload,
    LoraClassifierAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierAdapterStatePayload,
    PeftClassifierAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.registry import (
    get_shared_adapter_canonical_update_payload_format,
    get_shared_adapter_update_payload_formats,
    parse_shared_adapter_update_payload,
    register_shared_adapter_payload_family,
)


def test_shared_adapter_payloads_capture_revision_and_head_deltas(
    make_adapter_state_payload,
    make_adapter_update_payload,
) -> None:
    state = make_adapter_state_payload()
    delta = make_adapter_update_payload()

    assert state.label_weights["anxiety"] == [0.1, 0.2]
    assert state.adapter_kind == "classifier_head"
    assert delta.base_model_revision == "rev_001"
    assert delta.adapter_kind == "classifier_head"
    assert delta.label_counts["anxiety"] == 5


def test_generic_shared_adapter_loader_dispatches_classifier_head_fixture_payloads(
    tmp_path: Path,
    make_adapter_state_payload,
    make_adapter_update_payload,
) -> None:
    state_path = tmp_path / "state.json"
    update_path = tmp_path / "update.json"
    dump_shared_adapter_state_payload(
        state_path,
        make_adapter_state_payload(
            model_revision="rev_002",
            label_weights={"anxiety": [0.1, 0.2], "normal": [-0.1, -0.2]},
            label_biases={"anxiety": 0.0, "normal": 0.0},
        ),
    )
    dump_shared_adapter_update_payload(
        update_path,
        make_adapter_update_payload(
            base_model_revision="rev_002",
            label_weight_deltas={"anxiety": [0.01, 0.02]},
            label_bias_deltas={"anxiety": 0.01},
            example_count=3,
            mean_confidence=0.8,
            mean_margin=0.1,
        ),
    )

    loaded_state = load_shared_adapter_state_payload(state_path)
    loaded_update = load_shared_adapter_update_payload(update_path)

    assert isinstance(loaded_state, ClassifierHeadAdapterStatePayload)
    assert isinstance(loaded_update, ClassifierHeadAdapterUpdatePayload)
    assert loaded_state.adapter_kind == "classifier_head"
    assert loaded_update.adapter_kind == "classifier_head"


def test_shared_adapter_loader_accepts_registered_custom_family(
    tmp_path: Path,
    fixed_utc_time,
) -> None:
    class TestAdapterStatePayload(SharedAdapterStatePayload):
        bias: float

    class TestAdapterUpdatePayload(SharedAdapterUpdatePayload):
        shift: float

    register_shared_adapter_payload_family(
        "test_family",
        state_payload_type=TestAdapterStatePayload,
        update_payload_type=TestAdapterUpdatePayload,
    )

    state = TestAdapterStatePayload(
        schema_version="test_adapter_state.v1",
        adapter_kind="test_family",
        model_id="tracemind-embed",
        model_revision="rev_test",
        training_scope="adapter_only",
        updated_at=fixed_utc_time,
        bias=0.1,
    )
    update = TestAdapterUpdatePayload(
        schema_version="test_adapter_update.v1",
        adapter_kind="test_family",
        model_id="tracemind-embed",
        base_model_revision="rev_test",
        training_scope="adapter_only",
        example_count=2,
        created_at=fixed_utc_time,
        shift=0.05,
    )
    state_path = tmp_path / "custom_state.json"
    update_path = tmp_path / "custom_update.json"

    dump_shared_adapter_state_payload(state_path, state)
    dump_shared_adapter_update_payload(update_path, update)

    loaded_state = load_shared_adapter_state_payload(state_path)
    loaded_update = load_shared_adapter_update_payload(update_path)

    assert isinstance(loaded_state, TestAdapterStatePayload)
    assert isinstance(loaded_update, TestAdapterUpdatePayload)
    assert loaded_state.adapter_kind == "test_family"
    assert loaded_update.adapter_kind == "test_family"


def test_generic_shared_adapter_loader_dispatches_classifier_head_payloads(
    tmp_path: Path,
    fixed_utc_time,
) -> None:
    state_path = tmp_path / "classifier_state.json"
    update_path = tmp_path / "classifier_update.json"
    dump_shared_adapter_state_payload(
        state_path,
        ClassifierHeadAdapterStatePayload(
            adapter_kind="classifier_head",
            model_id="tracemind-embed",
            model_revision="rev_head_001",
            training_scope="head_only",
            updated_at=fixed_utc_time,
            label_weights={
                "anxiety": [0.5, -0.1],
                "normal": [-0.2, 0.4],
            },
            label_biases={"anxiety": 0.1, "normal": -0.1},
        ),
    )
    dump_shared_adapter_update_payload(
        update_path,
        ClassifierHeadAdapterUpdatePayload(
            adapter_kind="classifier_head",
            model_id="tracemind-embed",
            base_model_revision="rev_head_001",
            training_scope="head_only",
            example_count=3,
            created_at=fixed_utc_time,
            label_weight_deltas={
                "anxiety": [0.01, -0.02],
                "normal": [-0.01, 0.02],
            },
            label_bias_deltas={"anxiety": 0.03, "normal": -0.03},
            mean_confidence=0.92,
            mean_margin=0.4,
            label_counts={"anxiety": 2, "normal": 1},
        ),
    )

    loaded_state = load_shared_adapter_state_payload(state_path)
    loaded_update = load_shared_adapter_update_payload(update_path)

    assert isinstance(loaded_state, ClassifierHeadAdapterStatePayload)
    assert isinstance(loaded_update, ClassifierHeadAdapterUpdatePayload)
    assert loaded_state.embedding_dim == 2
    assert loaded_update.labels == ("anxiety", "normal")


def _lora_backbone_mapping() -> dict[str, object]:
    return {
        "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "backbone_revision": "main",
        "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "tokenizer_revision": "main",
        "pooling": "mean",
        "max_length": 256,
        "task_prefix": "",
    }


def _lora_config_mapping() -> dict[str, object]:
    return {
        "peft_adapter_name": "lora",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "bias": "none",
        "target_modules": "all-linear",
        "use_rslora": False,
    }


def _peft_adapter_config_mapping() -> dict[str, object]:
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


def test_generic_shared_adapter_loader_dispatches_lora_classifier_payloads(
    tmp_path: Path,
    fixed_utc_time,
) -> None:
    state_path = tmp_path / "lora_classifier_state.json"
    update_path = tmp_path / "lora_classifier_update.json"
    labels = ["anxiety", "depression", "normal", "suicidal"]
    dump_shared_adapter_state_payload(
        state_path,
        make_lora_classifier_state_payload(
            model_id="mxbai-lora-classifier",
            model_revision="rev_lora_001",
            training_scope="adapter_only",
            updated_at=fixed_utc_time,
            backbone=_lora_backbone_mapping(),
            lora_config=_lora_config_mapping(),
            label_schema=labels,
            lora_adapter_artifact_ref="server-artifact::lora/rev_lora_001",
            classifier_head_artifact_ref="server-artifact::head/rev_lora_001",
        ),
    )
    dump_shared_adapter_update_payload(
        update_path,
        make_lora_classifier_delta_payload(
            model_id="mxbai-lora-classifier",
            base_model_revision="rev_lora_001",
            training_scope="adapter_only",
            created_at=fixed_utc_time,
            backbone=_lora_backbone_mapping(),
            lora_config=_lora_config_mapping(),
            label_schema=labels,
            example_count=7,
            lora_delta_artifact_ref="client-update::lora/update_001",
            classifier_head_delta_artifact_ref="client-update::head/update_001",
            mean_confidence=0.91,
            mean_margin=0.32,
            label_counts={"anxiety": 4, "normal": 3},
            delta_l2_norm=1.25,
        ),
    )

    loaded_state = load_shared_adapter_state_payload(state_path)
    loaded_update = load_shared_adapter_update_payload(update_path)

    assert isinstance(loaded_state, LoraClassifierAdapterStatePayload)
    assert isinstance(loaded_update, LoraClassifierAdapterUpdatePayload)
    assert loaded_state.adapter_kind == "lora_classifier"
    assert loaded_state.labels == tuple(labels)
    assert loaded_update.base_model_revision == "rev_lora_001"
    assert loaded_update.labels == tuple(labels)
    assert loaded_update.l2_norm() == pytest.approx(1.25)


def test_generic_shared_adapter_loader_dispatches_peft_classifier_payloads(
    tmp_path: Path,
    fixed_utc_time,
) -> None:
    state_path = tmp_path / "peft_classifier_state.json"
    update_path = tmp_path / "peft_classifier_update.json"
    labels = ["anxiety", "depression", "normal", "suicidal"]
    dump_shared_adapter_state_payload(
        state_path,
        make_peft_classifier_state_payload(
            model_id="mxbai-peft-classifier",
            model_revision="rev_peft_001",
            training_scope="adapter_only",
            updated_at=fixed_utc_time,
            backbone=_lora_backbone_mapping(),
            peft_adapter_config=_peft_adapter_config_mapping(),
            label_schema=labels,
            peft_adapter_artifact_ref="server-artifact::peft/rev_peft_001",
            classifier_head_artifact_ref="server-artifact::head/rev_peft_001",
        ),
    )
    dump_shared_adapter_update_payload(
        update_path,
        make_peft_classifier_delta_payload(
            model_id="mxbai-peft-classifier",
            base_model_revision="rev_peft_001",
            training_scope="adapter_only",
            created_at=fixed_utc_time,
            backbone=_lora_backbone_mapping(),
            peft_adapter_config=_peft_adapter_config_mapping(),
            label_schema=labels,
            example_count=7,
            peft_adapter_delta_artifact_ref="client-update::peft/update_001",
            classifier_head_delta_artifact_ref="client-update::head/update_001",
            mean_confidence=0.91,
            mean_margin=0.32,
            label_counts={"anxiety": 4, "normal": 3},
            delta_l2_norm=1.25,
        ),
    )

    loaded_state = load_shared_adapter_state_payload(state_path)
    loaded_update = load_shared_adapter_update_payload(update_path)

    assert isinstance(loaded_state, PeftClassifierAdapterStatePayload)
    assert isinstance(loaded_update, PeftClassifierAdapterUpdatePayload)
    assert loaded_state.adapter_kind == "peft_classifier"
    assert loaded_state.peft_adapter_name == "lora"
    assert loaded_state.labels == tuple(labels)
    assert loaded_update.base_model_revision == "rev_peft_001"
    assert loaded_update.peft_adapter_name == "lora"
    assert loaded_update.labels == tuple(labels)
    assert loaded_update.l2_norm() == pytest.approx(1.25)


def test_lora_classifier_state_applies_identity_normalization_for_simulation(
    fixed_utc_time,
) -> None:
    state = make_lora_classifier_state_payload(
        model_id="mxbai-lora-classifier",
        model_revision="rev_lora_001",
        training_scope="adapter_only",
        updated_at=fixed_utc_time,
        backbone=_lora_backbone_mapping(),
        lora_config=_lora_config_mapping(),
        label_schema=["anxiety", "normal"],
    )

    assert state.apply([3.0, 4.0]) == pytest.approx([0.6, 0.8])

    with pytest.raises(ValueError, match="norm must be non-zero"):
        state.apply([0.0, 0.0])


def test_lora_classifier_update_supports_inline_delta_without_artifact_ref(
    fixed_utc_time,
) -> None:
    update = make_lora_classifier_delta_payload(
        model_id="mxbai-lora-classifier",
        base_model_revision="rev_lora_001",
        training_scope="adapter_only",
        created_at=fixed_utc_time,
        backbone=_lora_backbone_mapping(),
        lora_config=_lora_config_mapping(),
        label_schema=["anxiety", "normal"],
        example_count=2,
        lora_parameter_deltas={"backbone.q_proj.lora_A": [0.3, 0.4]},
        classifier_head_weight_deltas={
            "anxiety": [0.1, 0.2],
            "normal": [-0.1, -0.2],
        },
        classifier_head_bias_deltas={"anxiety": 0.05},
    )

    assert update.delta_format == "artifact_ref"
    assert update.classifier_head_bias_deltas == {
        "anxiety": 0.05,
        "normal": 0.0,
    }
    assert update.l2_norm() == pytest.approx(
        (0.3**2 + 0.4**2 + 0.1**2 + 0.2**2 + 0.1**2 + 0.2**2 + 0.05**2) ** 0.5
    )


def test_lora_classifier_update_supports_partitioned_delta_material(
    fixed_utc_time,
) -> None:
    update = make_lora_classifier_delta_payload(
        model_id="mxbai-lora-classifier",
        base_model_revision="rev_lora_001",
        training_scope="adapter_only",
        created_at=fixed_utc_time,
        backbone=_lora_backbone_mapping(),
        lora_config=_lora_config_mapping(),
        label_schema=["anxiety", "normal"],
        example_count=2,
        partitioned_deltas={
            "sigma": {
                "lora_parameter_deltas": {"backbone.q_proj.lora_A": [0.3, 0.4]},
                "classifier_head_weight_deltas": {
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                "classifier_head_bias_deltas": {"anxiety": 0.05},
            }
        },
        delta_format="partitioned_update",
    )

    assert set(update.partitioned_deltas or {}) == {"sigma"}
    assert (update.partitioned_deltas or {})["sigma"].classifier_head_bias_deltas == {
        "anxiety": 0.05,
        "normal": 0.0,
    }
    assert update.l2_norm() == pytest.approx(
        (0.3**2 + 0.4**2 + 0.1**2 + 0.2**2 + 0.1**2 + 0.2**2 + 0.05**2) ** 0.5
    )


def test_lora_classifier_update_supports_partitioned_delta_artifact_ref(
    fixed_utc_time,
) -> None:
    update = make_lora_classifier_delta_payload(
        model_id="mxbai-lora-classifier",
        base_model_revision="rev_lora_001",
        training_scope="adapter_only",
        created_at=fixed_utc_time,
        backbone=_lora_backbone_mapping(),
        lora_config=_lora_config_mapping(),
        label_schema=["anxiety", "normal"],
        example_count=2,
        partitioned_deltas_artifact_ref=(
            "aggregation_artifact::client_updates/round_0001/agent_01/"
            "update_001/partitioned_delta"
        ),
        delta_format="server_uploaded_artifact_ref",
        delta_l2_norm=1.25,
    )

    assert update.partitioned_deltas is None
    assert update.partitioned_deltas_artifact_ref is not None
    assert update.l2_norm() == pytest.approx(1.25)


def test_lora_classifier_update_requires_artifact_ref_or_inline_delta(
    fixed_utc_time,
) -> None:
    with pytest.raises(ValueError, match="artifact refs or inline deltas"):
        make_lora_classifier_delta_payload(
            model_id="mxbai-lora-classifier",
            base_model_revision="rev_lora_001",
            training_scope="adapter_only",
            created_at=fixed_utc_time,
            backbone=_lora_backbone_mapping(),
            lora_config=_lora_config_mapping(),
            label_schema=["anxiety", "normal"],
            example_count=2,
        )


def test_peft_classifier_update_supports_inline_delta_without_artifact_ref(
    fixed_utc_time,
) -> None:
    update = make_peft_classifier_delta_payload(
        model_id="mxbai-peft-classifier",
        base_model_revision="rev_peft_001",
        training_scope="adapter_only",
        created_at=fixed_utc_time,
        backbone=_lora_backbone_mapping(),
        peft_adapter_config=_peft_adapter_config_mapping(),
        label_schema=["anxiety", "normal"],
        example_count=2,
        peft_parameter_deltas={"backbone.q_proj.lora_A": [0.3, 0.4]},
        classifier_head_weight_deltas={
            "anxiety": [0.1, 0.2],
            "normal": [-0.1, -0.2],
        },
        classifier_head_bias_deltas={"anxiety": 0.05},
    )

    assert update.delta_format == "artifact_ref"
    assert update.classifier_head_bias_deltas == {
        "anxiety": 0.05,
        "normal": 0.0,
    }
    assert update.l2_norm() == pytest.approx(
        (0.3**2 + 0.4**2 + 0.1**2 + 0.2**2 + 0.1**2 + 0.2**2 + 0.05**2) ** 0.5
    )


def test_peft_classifier_update_supports_partitioned_delta_material(
    fixed_utc_time,
) -> None:
    update = make_peft_classifier_delta_payload(
        model_id="mxbai-peft-classifier",
        base_model_revision="rev_peft_001",
        training_scope="adapter_only",
        created_at=fixed_utc_time,
        backbone=_lora_backbone_mapping(),
        peft_adapter_config=_peft_adapter_config_mapping(),
        label_schema=["anxiety", "normal"],
        example_count=2,
        partitioned_deltas={
            "sigma": {
                "peft_parameter_deltas": {"backbone.q_proj.lora_A": [0.3, 0.4]},
                "classifier_head_weight_deltas": {
                    "anxiety": [0.1, 0.2],
                    "normal": [-0.1, -0.2],
                },
                "classifier_head_bias_deltas": {"anxiety": 0.05},
            }
        },
        delta_format="partitioned_update",
    )

    assert set(update.partitioned_deltas or {}) == {"sigma"}
    assert (update.partitioned_deltas or {})["sigma"].classifier_head_bias_deltas == {
        "anxiety": 0.05,
        "normal": 0.0,
    }
    assert update.l2_norm() == pytest.approx(
        (0.3**2 + 0.4**2 + 0.1**2 + 0.2**2 + 0.1**2 + 0.2**2 + 0.05**2) ** 0.5
    )


def test_payload_registry_exposes_lora_classifier_update_formats() -> None:
    assert (
        get_shared_adapter_canonical_update_payload_format("lora_classifier")
        == LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
    )
    assert get_shared_adapter_update_payload_formats("lora_classifier") == (
        LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    )


def test_payload_registry_exposes_peft_classifier_update_formats() -> None:
    assert (
        get_shared_adapter_canonical_update_payload_format("peft_classifier")
        == PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
    )
    assert get_shared_adapter_update_payload_formats("peft_classifier") == (
        PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    )


def test_adapter_registry_requires_adapter_kind_without_legacy_vector_schema() -> None:
    with pytest.raises(ValueError, match="adapter_kind"):
        parse_shared_adapter_update_payload(
            {
                "schema_version": "custom_adapter_delta.v1",
                "model_id": "model",
                "base_model_revision": "rev_1",
                "training_scope": "adapter_only",
                "example_count": 1,
            }
        )


def test_adapter_registry_rejects_legacy_vector_schema_without_adapter_kind() -> None:
    with pytest.raises(ValueError, match="adapter_kind"):
        parse_shared_adapter_update_payload(
            {
                "schema_version": "vector_adapter_delta.v1",
                "model_id": "model",
                "base_model_revision": "rev_1",
                "training_scope": "adapter_only",
                "dimension_deltas": [0.1],
                "example_count": 1,
                "mean_confidence": 0.8,
            }
        )
