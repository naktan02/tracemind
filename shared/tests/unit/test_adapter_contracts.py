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
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.io import (
    dump_shared_adapter_state_payload,
    dump_shared_adapter_update_payload,
    load_shared_adapter_state_payload,
    load_shared_adapter_update_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierAdapterStatePayload,
    LoraClassifierAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.registry import (
    get_shared_adapter_canonical_update_payload_format,
    get_shared_adapter_update_payload_formats,
    register_shared_adapter_payload_family,
)
from shared.src.contracts.training_contracts import UpdatePayloadFormat


def test_shared_adapter_payloads_capture_revision_and_scales(
    make_adapter_state_payload,
    make_adapter_update_payload,
) -> None:
    state = make_adapter_state_payload()
    delta = make_adapter_update_payload()

    assert state.dimension_scales[1] == 0.95
    assert state.adapter_kind == "diagonal_scale"
    assert delta.base_model_revision == "rev_001"
    assert delta.adapter_kind == "diagonal_scale"
    assert delta.label_counts["anxiety"] == 5


def test_generic_shared_adapter_loader_dispatches_diagonal_scale_payloads(
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
            dimension_scales=[1.0, 1.1],
        ),
    )
    dump_shared_adapter_update_payload(
        update_path,
        make_adapter_update_payload(
            base_model_revision="rev_002",
            dimension_deltas=[0.01, 0.02],
            example_count=3,
            mean_confidence=0.8,
            mean_margin=0.1,
        ),
    )

    loaded_state = load_shared_adapter_state_payload(state_path)
    loaded_update = load_shared_adapter_update_payload(update_path)

    assert isinstance(loaded_state, DiagonalScaleAdapterStatePayload)
    assert isinstance(loaded_update, DiagonalScaleAdapterUpdatePayload)
    assert loaded_state.adapter_kind == "diagonal_scale"
    assert loaded_update.adapter_kind == "diagonal_scale"


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
        adapter_kind="test_family",
        model_id="tracemind-embed",
        model_revision="rev_test",
        training_scope="adapter_only",
        updated_at=fixed_utc_time,
        bias=0.1,
    )
    update = TestAdapterUpdatePayload(
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


def test_payload_registry_exposes_lora_classifier_update_formats() -> None:
    assert (
        get_shared_adapter_canonical_update_payload_format("lora_classifier")
        == UpdatePayloadFormat.LORA_CLASSIFIER_UPDATE.value
    )
    assert get_shared_adapter_update_payload_formats("lora_classifier") == (
        "lora_classifier_update",
    )
