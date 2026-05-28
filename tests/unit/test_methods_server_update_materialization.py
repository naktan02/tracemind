"""Server update materialization preflight tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from methods.adaptation.peft_text_classifier.config import (
    PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
)
from methods.adaptation.server_update_materialization import (
    require_server_materializable_update_payload,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_peft_classifier_delta_payload,
)


def test_unknown_adapter_kind_materialization_defaults_to_noop() -> None:
    payload = SharedAdapterUpdatePayload(
        schema_version="unit_test_adapter_update.v1",
        adapter_kind="unit_test_adapter",
        model_id="tracemind-test",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        example_count=1,
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )

    require_server_materializable_update_payload(payload)


def test_lora_materialization_rejects_agent_local_refs_without_inline_delta() -> None:
    payload = _build_lora_payload(
        lora_delta_artifact_ref="agent-local://agent_001/lora_delta",
        classifier_head_delta_artifact_ref=(
            "agent-local://agent_001/classifier_head_delta"
        ),
    )

    with pytest.raises(ValueError, match="agent-local artifact"):
        require_server_materializable_update_payload(payload)


def test_lora_materialization_allows_agent_local_refs_with_inline_delta() -> None:
    payload = _build_lora_payload(
        lora_delta_artifact_ref="agent-local://agent_001/lora_delta",
        classifier_head_delta_artifact_ref=(
            "agent-local://agent_001/classifier_head_delta"
        ),
        lora_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.1]},
        classifier_head_weight_deltas={
            "anxiety": [0.1, 0.2],
            "normal": [-0.1, -0.2],
        },
    )

    require_server_materializable_update_payload(payload)


def test_lora_materialization_allows_server_owned_refs_without_inline_delta() -> None:
    payload = _build_lora_payload(
        lora_delta_artifact_ref="aggregation_artifact::agent_001/lora_delta",
        classifier_head_delta_artifact_ref=(
            "aggregation_artifact::agent_001/classifier_head_delta"
        ),
        delta_format=PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED,
    )

    require_server_materializable_update_payload(payload)


def test_peft_materialization_rejects_local_refs_without_inline_delta() -> None:
    payload = _build_peft_payload(
        peft_adapter_delta_artifact_ref="agent-local://agent_001/peft_delta",
        classifier_head_delta_artifact_ref=(
            "agent-local://agent_001/classifier_head_delta"
        ),
    )

    with pytest.raises(ValueError, match="agent-local artifact"):
        require_server_materializable_update_payload(payload)


def test_peft_classifier_materialization_allows_inline_delta_with_local_refs() -> None:
    payload = _build_peft_payload(
        peft_adapter_delta_artifact_ref="agent-local://agent_001/peft_delta",
        classifier_head_delta_artifact_ref=(
            "agent-local://agent_001/classifier_head_delta"
        ),
        peft_parameter_deltas={"encoder.q_proj.lora_A": [0.1, -0.1]},
        classifier_head_weight_deltas={
            "anxiety": [0.1, 0.2],
            "normal": [-0.1, -0.2],
        },
    )

    require_server_materializable_update_payload(payload)


def test_lora_classifier_materialization_rejects_mismatched_payload_type() -> None:
    payload = SharedAdapterUpdatePayload(
        schema_version="lora_classifier_delta.v1",
        adapter_kind="lora_classifier",
        model_id="tracemind-test",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        example_count=1,
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="PEFT classifier delta"):
        require_server_materializable_update_payload(payload)


def _build_lora_payload(**overrides):
    defaults = {
        "model_id": "tracemind-peft",
        "base_model_revision": "rev_000",
        "training_scope": "adapter_only",
        "backbone": {
            "backbone_model_id": "mxbai",
            "backbone_revision": "main",
            "tokenizer_model_id": "mxbai",
            "tokenizer_revision": "main",
            "pooling": "mean",
            "max_length": 256,
            "task_prefix": "",
        },
        "lora_config": {
            "peft_adapter_name": "lora",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": "all-linear",
            "use_rslora": False,
        },
        "label_schema": ["anxiety", "normal"],
        "example_count": 2,
        "mean_confidence": 0.8,
        "mean_margin": 0.2,
    }
    defaults.update(overrides)
    return make_lora_classifier_delta_payload(**defaults)


def _build_peft_payload(**overrides):
    defaults = {
        "model_id": "tracemind-peft",
        "base_model_revision": "rev_000",
        "training_scope": "adapter_only",
        "backbone": {
            "backbone_model_id": "mxbai",
            "backbone_revision": "main",
            "tokenizer_model_id": "mxbai",
            "tokenizer_revision": "main",
            "pooling": "mean",
            "max_length": 256,
            "task_prefix": "",
        },
        "peft_adapter_config": {
            "peft_adapter_name": "lora",
            "parameters": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "bias": "none",
                "target_modules": "all-linear",
                "use_rslora": False,
            },
        },
        "label_schema": ["anxiety", "normal"],
        "example_count": 2,
        "mean_confidence": 0.8,
        "mean_margin": 0.2,
    }
    defaults.update(overrides)
    return make_peft_classifier_delta_payload(**defaults)
