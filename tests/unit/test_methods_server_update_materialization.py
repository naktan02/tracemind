"""Server update materialization preflight tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from methods.adaptation.server_update_materialization import (
    require_server_materializable_update_payload,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
)


def test_unknown_adapter_kind_materialization_defaults_to_noop() -> None:
    payload = SharedAdapterUpdatePayload(
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


def test_lora_classifier_materialization_rejects_mismatched_payload_type() -> None:
    payload = SharedAdapterUpdatePayload(
        adapter_kind="lora_classifier",
        model_id="tracemind-test",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        example_count=1,
        created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="LoraClassifierDelta"):
        require_server_materializable_update_payload(payload)


def _build_lora_payload(**overrides):
    defaults = {
        "model_id": "tracemind-lora",
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
